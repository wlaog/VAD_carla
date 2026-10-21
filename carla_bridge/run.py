import math
import random
import carla, pygame, queue, os, cv2, pickle, numpy as np, sys
from vad_carla_bridge import VADAdapter, VADAgent, PIDController

# --- 配置 ---
# HOST = '172.22.176.1'; PORT = 2000; TM_PORT = 8000
# HOST = '192.168.0.113'; PORT = 2000; TM_PORT = 8000
HOST = '127.0.0.1'; PORT = 2000; TM_PORT = 8000
CONFIG = 'projects/configs/VAD/VAD_tiny_e2e.py'; CKPT = 'ckpts/VAD_tiny.pth'
SAVE_DIR = 'carla_bridge/InputDict_check/input_dictC'
IMAGE_SAVE_DIR = os.path.join(SAVE_DIR, 'images')
for d in [SAVE_DIR, IMAGE_SAVE_DIR]: os.makedirs(d, exist_ok=True)

# --- 随机车辆配置 ---
NUM_VEHICLES = 30  # 生成的车辆数量
NUM_WALKERS = 10   # 生成的行人数量（可选）

# --- 颜色与 UI ---
WHITE = (255, 255, 255); GREEN = (0, 255, 0); RED = (255, 0, 0); BLACK = (0, 0, 0); GREY = (30, 30, 30); YELLOW = (255, 255, 0)

def spawn_vehicles(world, client, tm_port, num_vehicles):
    """
    生成随机车辆
    返回生成的车辆列表
    """
    vehicles = []
    
    # 获取车辆蓝图
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')
    
    # 过滤掉自行车等（可选）
    vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) >= 4]
    
    # 获取所有出生点
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    # Traffic Manager
    tm = client.get_trafficmanager(tm_port)
    
    print(f"Spawning {num_vehicles} vehicles...")
    
    for i, spawn_point in enumerate(spawn_points[:num_vehicles]):
        bp = random.choice(vehicle_bps)
        
        # 设置随机颜色（如果可用）
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        # 设置为非自动驾驶（由 Traffic Manager 控制）
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', 'autopilot')
        
        try:
            vehicle = world.spawn_actor(bp, spawn_point)
            vehicle.set_autopilot(True, tm_port)
            vehicles.append(vehicle)
        except Exception as e:
            print(f"Failed to spawn vehicle {i}: {e}")
            continue
    
    print(f"Successfully spawned {len(vehicles)} vehicles")
    return vehicles

def spawn_walkers(world, client, num_walkers):
    """
    生成随机行人
    返回生成的行人和控制器列表
    """
    walkers = []
    walker_controllers = []
    
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    walker_controller_bp = bp_lib.find('controller.ai.walker')
    
    # 获取行人出生点
    spawn_points = []
    for _ in range(num_walkers * 2):  # 多取一些以防失败
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_points.append(carla.Transform(loc))
    
    print(f"Spawning {num_walkers} walkers...")
    
    for i, spawn_point in enumerate(spawn_points[:num_walkers]):
        bp = random.choice(walker_bps)
        
        # 设置为不可被撞倒（可选）
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        
        try:
            walker = world.spawn_actor(bp, spawn_point)
            walkers.append(walker)
            
            # 生成控制器
            controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
            walker_controllers.append(controller)
        except Exception as e:
            print(f"Failed to spawn walker {i}: {e}")
            continue
    
    # 启动行人控制器
    world.tick()  # 确保控制器已创建
    for controller in walker_controllers:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1 + random.random())  # 随机速度 1-2 m/s
    
    print(f"Successfully spawned {len(walkers)} walkers")
    return walkers, walker_controllers

def destroy_actors(actors):
    """销毁 actor 列表"""
    for actor in actors:
        if actor is not None and actor.is_alive:
            actor.destroy()

def draw_ui(display, mode, control, speed, step, num_vehicles=0):
    display.fill(GREY)
    font = pygame.font.SysFont("Arial", 18)
    # 状态文本
    txt = [
        f"MODE: {mode} (M:Manual, A:Auto, V:VAD, Q:Quit)",
        f"SPEED: {speed:.1f} km/h",
        f"STEP: {step}",
        f"Traffic Vehicles: {num_vehicles}"
    ]
    for i, t in enumerate(txt): display.blit(font.render(t, True, WHITE), (20, 20 + i*25))
    
    # 绘制方向盘
    center = (500, 300); radius = 50
    pygame.draw.circle(display, WHITE, center, radius, 2)
    end_x = center[0] + radius * math.sin(control.steer * math.pi/2)
    end_y = center[1] - radius * math.cos(control.steer * math.pi/2)
    pygame.draw.line(display, GREEN, center, (end_x, end_y), 4)
    
    # 绘制踏板 (油门/刹车)
    pygame.draw.rect(display, WHITE, (450, 100, 20, 100), 2)
    pygame.draw.rect(display, GREEN, (450, 200 - control.throttle*100, 20, control.throttle*100))
    pygame.draw.rect(display, WHITE, (480, 100, 20, 100), 2)
    pygame.draw.rect(display, RED, (480, 200 - control.brake*100, 20, control.brake*100))
    display.blit(font.render("T", True, WHITE), (455, 205))
    display.blit(font.render("B", True, WHITE), (485, 205))

def main():
    pygame.init()
    display = pygame.display.set_mode((600, 400))
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.get_world()

    # 天气设置
    weather = world.get_weather()
    weather.cloudiness = 50.0
    weather.sun_altitude_angle = 25.0
    weather.sun_azimuth_angle = 0.0
    weather.precipitation_deposits = 8.0
    weather.fog_density = 5.0
    world.set_weather(weather)
    
    # 同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    
    # 设置 Traffic Manager 行为（可选调整）
    tm.set_global_distance_to_leading_vehicle(2.5)  # 跟车距离
    # tm.set_hybrid_physics_mode(True)  # 混合物理模式（远处车辆简化物理）
    # tm.set_hybrid_physics_radius(50.0)

    # --- 生成 Ego 车辆 ---
    bp = world.get_blueprint_library().filter('vehicle.audi.tt')[0]
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn = random.choice(spawn_points)
    ego = world.spawn_actor(bp, ego_spawn)
    
    # --- 生成随机交通车辆 ---
    # 排除 ego 使用的出生点
    traffic_vehicles = spawn_vehicles(world, client, TM_PORT, NUM_VEHICLES)
    
    # --- 生成行人（可选） ---
    walkers, walker_controllers = [], []
    if NUM_WALKERS > 0:
        walkers, walker_controllers = spawn_walkers(world, client, NUM_WALKERS)
    
    # --- 摄像头设置 ---
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '1600')
    cam_bp.set_attribute('image_size_y', '900')

    configs = [
        (1.72, 0.00, 1.49, 0.0, 1252.8),    # CAM_FRONT
        (1.58, 0.50, 1.52, 55.0, 1252.7),   # CAM_FRONT_RIGHT
        (1.58, -0.50, 1.51, -55.0, 1257.9), # CAM_FRONT_LEFT
        (0.06, -0.01, 1.57, 180.0, 796.9),  # CAM_BACK
        (1.05, -0.48, 1.56, -118.0, 1255.0),# CAM_BACK_LEFT
        (1.06, 0.47, 1.55, 113.0, 1250.0)   # CAM_BACK_RIGHT
    ]
    
    cams, qs = [], []
    for x, y, z, yaw, fx in configs:
        fov = 2*math.atan(1600/(2*fx)) * 180/math.pi
        cam_bp.set_attribute('fov', str(fov))
        c = world.spawn_actor(
            cam_bp, 
            carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)), 
            attach_to=ego
        )
        q = queue.Queue()
        c.listen(q.put)
        cams.append(c)
        qs.append(q)

    # 初始化大脑
    sys.path.append('.')
    import projects.mmdet3d_plugin
    adapter = VADAdapter(ego)
    agent = VADAgent(CONFIG, CKPT)
    pid = PIDController()
    spec = world.get_spectator()
    step_idx = 0
    mode = 'PAUSE'
    cur_ctrl = carla.VehicleControl()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        mode = 'AUTO'
                    if event.key == pygame.K_m:
                        mode = 'MANUAL'
                    if event.key == pygame.K_a:
                        mode = 'AUTO'
                    if event.key == pygame.K_v:
                        mode = 'VAD'
                    if event.key == pygame.K_q:
                        return
                    # 按 R 重新生成交通
                    if event.key == pygame.K_r:
                        print("Respawning traffic...")
                        destroy_actors(traffic_vehicles)
                        traffic_vehicles = spawn_vehicles(world, client, TM_PORT, NUM_VEHICLES)

            if mode != 'PAUSE':
                world.tick()
                imgs = [q.get() for q in qs]
                img_paths = []
                p_dir = os.path.join(IMAGE_SAVE_DIR, f'step_{step_idx:04d}')
                os.makedirs(p_dir, exist_ok=True)
                for i, im in enumerate(imgs):
                    p = os.path.join(p_dir, f'cam_{i}.jpg')
                    img_paths.append(p)
                    cv2.imwrite(p, np.reshape(np.frombuffer(im.raw_data, "uint8"), (900, 1600, 4))[:,:,:3])
                
                input_dict = adapter.make_input_dict(
                    step_idx, 
                    world.get_snapshot().timestamp.elapsed_seconds, 
                    img_paths, 
                    cams
                )
                with open(os.path.join(SAVE_DIR, f'step_{step_idx:04d}.pkl'), 'wb') as f:
                    pickle.dump(input_dict, f)
                
                # 模式控制逻辑
                if mode == 'AUTO':
                    ego.set_autopilot(True, TM_PORT)
                    cur_ctrl = ego.get_control()
                elif mode == 'MANUAL':
                    ego.set_autopilot(False)
                    pygame.event.pump()
                    keys = pygame.key.get_pressed()
                    
                    v = ego.get_velocity()
                    fwd = ego.get_transform().get_forward_vector()
                    speed_signed = v.x * fwd.x + v.y * fwd.y + v.z * fwd.z
                    speed_kmh = abs(speed_signed) * 3.6
                    
                    if keys[pygame.K_UP]:
                        if cur_ctrl.reverse and speed_kmh > 0.5:
                            cur_ctrl.throttle = 0.0
                            cur_ctrl.brake = min(1.0, cur_ctrl.brake + 0.15)
                        else:
                            cur_ctrl.reverse = False
                            cur_ctrl.throttle = min(1.0, cur_ctrl.throttle + 0.1)
                            cur_ctrl.brake = 0.0
                    elif keys[pygame.K_DOWN]:
                        if not cur_ctrl.reverse and speed_kmh > 0.5:
                            cur_ctrl.throttle = 0.0
                            cur_ctrl.brake = min(1.0, cur_ctrl.brake + 0.15)
                        else:
                            cur_ctrl.reverse = True
                            cur_ctrl.throttle = min(0.6, cur_ctrl.throttle + 0.08)
                            cur_ctrl.brake = 0.0
                    else:
                        cur_ctrl.throttle = max(0.0, cur_ctrl.throttle - 0.05)
                        cur_ctrl.brake = max(0.0, cur_ctrl.brake - 0.1)
                    
                    if keys[pygame.K_LEFT]:
                        cur_ctrl.steer = max(-1.0, cur_ctrl.steer - 0.08)
                    elif keys[pygame.K_RIGHT]:
                        cur_ctrl.steer = min(1.0, cur_ctrl.steer + 0.08)
                    else:
                        if abs(cur_ctrl.steer) < 0.05:
                            cur_ctrl.steer = 0.0
                        else:
                            cur_ctrl.steer *= 0.85
                    
                    cur_ctrl.hand_brake = keys[pygame.K_SPACE]
                    ego.apply_control(cur_ctrl)
                    
                elif mode == 'VAD':
                    ego.set_autopilot(False)
                    traj = agent.run_step(input_dict)
                    traj = traj[:, [1, 0]]
                    cur_ctrl = pid.get_control(
                        traj, 
                        math.sqrt(ego.get_velocity().x**2 + ego.get_velocity().y**2)
                    )
                    ego.apply_control(cur_ctrl)
                
                step_idx += 1

            # 更新视角与UI
            t = ego.get_transform()
            spec.set_transform(carla.Transform(
                t.location - 6*t.get_forward_vector() + carla.Location(z=2.5), 
                t.rotation
            ))
            speed = 3.6 * math.sqrt(ego.get_velocity().x**2 + ego.get_velocity().y**2)
            draw_ui(display, mode, cur_ctrl, speed, step_idx, len(traffic_vehicles))
            pygame.display.flip()

    finally:
        print("Cleaning up...")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        # 销毁所有 actors
        destroy_actors(walker_controllers)
        destroy_actors(walkers)
        destroy_actors(traffic_vehicles)
        destroy_actors(cams)
        if ego is not None:
            ego.destroy()
        
        pygame.quit()

if __name__ == '__main__':
    main()