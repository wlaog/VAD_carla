import math
import json
import os
import sys
import numpy as np
import carla
import random
import queue
import pygame
import cv2
import pickle
from pyquaternion import Quaternion
from collections import deque
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT, K_t, K_a, K_q, K_r, K_RETURN

# --- 路径与配置 (保持您的) ---
sys.path.append('/home/wjc/carla_source/carla')
sys.path.append('./TCP')

HOST = '172.22.176.1'; PORT = 2000; TM_PORT = 8000
MAP_NAME = 'Town04'; IMG_WIDTH = 800; IMG_HEIGHT = 450
SAVE_DIR = 'carla_bridge/step_debug_info'
IMAGE_SAVE_DIR = os.path.join(SAVE_DIR, 'images')
if not os.path.exists(IMAGE_SAVE_DIR): os.makedirs(IMAGE_SAVE_DIR)

# --- 颜色定义 (保留您的) ---
GREY = (50, 50, 50); WHITE = (255, 255, 255); GREEN = (0, 255, 0)
RED = (255, 50, 50); BLUE = (50, 50, 255); YELLOW = (255, 255, 0); BLACK = (0, 0, 0)

# --- VAD 数据适配类 (全面补全版) ---
class VADAdapter:
    def __init__(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle
        self.frame_idx = 0
        # 1. 历史轨迹缓冲区 (存10帧，对应 nuScenes 2秒)
        self.his_buffer = deque(maxlen=10)
        self.last_yaw = None
        
        # 内参 (4x4 格式)
        f = IMG_WIDTH / (2 * math.tan(math.radians(50)))
        self.intrinsic = np.array([
            [f, 0, 400, 0],
            [0, f, 225, 0],
            [0, 0, 1,   0],
            [0, 0, 0,   1]
        ])

    def get_can_bus(self, timestamp):
        t = self.ego_vehicle.get_transform()
        v = self.ego_vehicle.get_velocity()
        a = self.ego_vehicle.get_acceleration()
        rot = t.rotation
        
        # 坐标与旋转纠偏 (LHS -> RHS)
        pos_x, pos_y, pos_z = t.location.x, -t.location.y, t.location.z
        vel_x, vel_y, vel_z = v.x, -v.y, v.z
        acc_x, acc_y, acc_z = a.x, -a.y, a.z
        yaw_nusc = -rot.yaw
        yaw_rad = math.radians(yaw_nusc)

        # 记录历史位置 (相对于当前位置，用于 ego_his_trajs)
        # 这里存的是全局右手系坐标
        self.his_buffer.append(np.array([pos_x, pos_y]))

        # 计算 Yaw Rate
        yaw_rate = 0
        if self.last_yaw is not None:
            yaw_rate = (yaw_rad - self.last_yaw) / 0.1
        self.last_yaw = yaw_rad

        q = Quaternion(axis=[0, 0, 1], radians=yaw_rad)

        can_bus = np.zeros(18)
        can_bus[0:3] = [pos_x, pos_y, pos_z]           # [0-2] 全局坐标
        can_bus[3:7] = [q.w, q.x, q.y, q.z]            # [3-6] 四元数
        can_bus[7:10] = [vel_x, vel_y, vel_z]          # [7-9] 全局速度
        can_bus[10:13] = [acc_x, acc_y, acc_z]         # [10-12] 全局加速度
        can_bus[13] = math.sqrt(vel_x**2 + vel_y**2)   # [13] 标量速度
        can_bus[15] = yaw_rate                         # [15] 偏航角速度
        can_bus[16] = yaw_rad                          # [16] Yaw弧度
        can_bus[17] = yaw_nusc % 360                   # [17] Yaw角度
        return can_bus

    def get_ego_lcf_feat(self):
        """补全 ego_lcf_feat (9维)"""
        v = self.ego_vehicle.get_velocity()
        # 简单填充：速度、加速度、尺寸等
        return np.array([v.x, v.y, 0, 4.5, 2.0, 0, 0, v.x, v.y], dtype=np.float32)

    def get_lidar2global(self):
        """补全 lidar2global (4x4)"""
        # 因为我们假设 LiDAR 在 Ego 中心，所以直接拿 Ego 的矩阵
        matrix = np.array(self.ego_vehicle.get_transform().get_matrix())
        # Y轴修正
        matrix[0:3, 1] *= -1 
        return matrix

    def get_ego_his_trajs(self):
        """补全 ego_his_trajs (10, 2)"""
        if len(self.his_buffer) < 10:
            return np.zeros((10, 2), dtype=np.float32)
        
        curr_pos = self.his_buffer[-1]
        # 计算相对于当前点的历史位移
        trajs = np.array([p - curr_pos for p in self.his_buffer], dtype=np.float32)
        return trajs

    def compute_lidar2img(self, cam_actor):
        world_2_cam = np.array(cam_actor.get_transform().get_inverse_matrix())
        ego_2_world = np.array(self.ego_vehicle.get_transform().get_matrix())
        l2c_carla = world_2_cam @ ego_2_world
        
        flip_y = np.eye(4); flip_y[1, 1] = -1
        c_to_v = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        
        return (self.intrinsic @ c_to_v @ l2c_carla @ flip_y).tolist()

    def make_input_dict(self, step_idx, timestamp, img_paths, cam_actors):
        """组装全量 input_dict"""
        t = self.ego_vehicle.get_transform()
        q = Quaternion(axis=[0, 0, 1], degrees=-t.rotation.yaw)
        
        input_dict = {
            # 1. 基础索引与路径
            'sample_idx': f"STEP_{step_idx:04d}",
            'pts_filename': "dummy_path",
            'sweeps': [],
            'scene_token': "carla_scene_001",
            'frame_idx': step_idx,
            'timestamp': timestamp,
            
            # 2. 全局位姿
            'ego2global_translation': [t.location.x, -t.location.y, t.location.z],
            'ego2global_rotation': [q.w, q.x, q.y, q.z],
            'lidar2global': self.get_lidar2global(),
            
            # 3. 传感器变换
            'lidar2ego': np.eye(4), # 假设重合
            'lidar2ego_translation': [0, 0, 0],
            'lidar2ego_rotation': [1, 0, 0, 0],
            'camera2ego': [np.eye(4) for _ in range(6)], # 简化处理
            'camera_intrinsics': [self.intrinsic for _ in range(6)],
            'lidar2cam': [np.eye(4) for _ in range(6)],
            'lidar2img': [self.compute_lidar2img(c) for c in cam_actors],
            'cam_intrinsic': [self.intrinsic for _ in range(6)],
            
            # 4. 车辆状态
            'can_bus': self.get_can_bus(timestamp),
            'ego_his_trajs': self.get_ego_his_trajs(),
            'ego_lcf_feat': self.get_ego_lcf_feat(),
            
            # 5. 推理/评估辅助 (伪造或真值)
            'img_filename': img_paths,
            'map_location': "singapore-onenorth", # VAD 需要一个存在的地图名
            'fut_valid_flag': True,
            'ego_fut_trajs': np.zeros((6, 2)), # 规划答案，推理不需要
            'ego_fut_masks': np.ones(6),
            'ego_fut_cmd': np.array([0, 0, 1]), # 直行命令
            'ann_info': {'gt_bboxes_3d': [], 'gt_labels_3d': []},
            'prev_idx': None,
            'next_idx': None
        }
        return input_dict

# --- 辅助：JSON 序列化工具 ---
def save_step_json(step_idx, data_dict):
    def default_converter(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)
    
    file_path = os.path.join(SAVE_DIR, f'step_{step_idx:04d}.json')
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=4, default=default_converter)
    print(f"✅ Step {step_idx} 数据已保存至: {file_path}")
    
def save_step_pkl(step_idx, data_dict):
    file_path = os.path.join(SAVE_DIR, f'step_{step_idx:04d}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"✅ Step {step_idx} 数据已保存至: {file_path}")

# --- 核心函数 (保留您的 UI 逻辑) ---
def init_dashboard():
    pygame.init()
    display = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("VAD Step Debugger")
    return display, pygame.font.SysFont("monospace", 50, bold=True), pygame.font.SysFont("monospace", 20)

def draw_dashboard(display, font_large, font_small, vehicle, control_mode, last_control, step_idx):
    display.fill(GREY)
    v = vehicle.get_velocity()
    speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    speed_text = font_large.render(f"{int(speed_kmh)}", True, WHITE)
    display.blit(speed_text, (300 - speed_text.get_width()//2, 50))
    
    step_text = font_small.render(f"STEP: {step_idx} (PAUSED - Press ENTER)", True, YELLOW)
    display.blit(step_text, (20, 360))

def main():
    actor_list = []; npc_list = []; cam_actors = []
    display, font_large, font_small = init_dashboard()
    
    try:
        client = carla.Client(HOST, PORT); client.set_timeout(20.0)
        world = client.load_world(MAP_NAME) if MAP_NAME not in client.get_world().get_map().name else client.get_world()
        
        # 同步模式配置
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        
        # 1. 生成交通 (保留您的)
        tm = client.get_trafficmanager(TM_PORT); tm.set_synchronous_mode(True)
        # npc_list = spawn_traffic(world, client, tm, NUMBER_OF_VEHICLES) # 奶奶可自行开启

        # 2. 生成主角车 (保留您的 Audi TT)
        bp_lib = world.get_blueprint_library()
        ego_vehicle = world.spawn_actor(bp_lib.filter('vehicle.audi.tt')[0], random.choice(world.get_map().get_spawn_points()))
        actor_list.append(ego_vehicle); ego_vehicle.set_autopilot(True, TM_PORT)

        # 3. 挂载 6 路相机 (VAD 核心)
        cam_configs = [
            (1.2, 0, 1.6, 0), (0.5, 0.7, 1.6, 55), (0.5, -0.7, 1.6, -55),
            (-1.2, 0, 1.6, 180), (-0.5, -0.7, 1.6, -110), (-0.5, 0.7, 1.6, 110)
        ]
        cam_queues = []
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMG_WIDTH)); cam_bp.set_attribute('image_size_y', str(IMG_HEIGHT))

        for i, (x, y, z, yaw) in enumerate(cam_configs):
            cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)), attach_to=ego_vehicle)
            q = queue.Queue(); cam.listen(q.put)
            cam_actors.append(cam); cam_queues.append(q); actor_list.append(cam)

        adapter = VADAdapter(ego_vehicle)
        spectator = world.get_spectator()
        step_idx = 0
        is_paused = True

        # --- 核心循环 ---
        while True:
            if not is_paused:
                world.tick()
                imgs_raw = [q.get() for q in cam_queues]
                img_paths = []
                current_step_dir = os.path.join(IMAGE_SAVE_DIR, f'step_{step_idx:04d}')
                if not os.path.exists(current_step_dir): os.makedirs(current_step_dir)

                for i, img_obj in enumerate(imgs_raw):
                    array = np.frombuffer(img_obj.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (img_obj.height, img_obj.width, 4))
                    img_bgr = array[:, :, :3]
                    path = os.path.join(current_step_dir, f'cam_{i}.jpg')
                    cv2.imwrite(path, img_bgr); img_paths.append(path)

                # --- 生成全量字典 ---
                timestamp = world.get_snapshot().timestamp.elapsed_seconds
                input_dict = adapter.make_input_dict(step_idx, timestamp, img_paths, cam_actors)
                
                save_step_json(step_idx, input_dict)
                save_step_pkl(step_idx, input_dict)
                print(f"📦 Step {step_idx}: 全量数据包已封装完毕。")
                # is_paused = True; 
                step_idx += 1

            # --- A. 视角跟随 (保留您的逻辑) ---
            ego_trans = ego_vehicle.get_transform()
            loc = ego_trans.location - 6 * ego_trans.get_forward_vector() + carla.Location(z=2.5)
            spectator.set_transform(carla.Transform(loc, ego_trans.rotation))

            # --- B. 事件处理 ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return
                if event.type == pygame.KEYDOWN:
                    if event.key == K_RETURN: is_paused = False # 回车走一步
                    if event.key == K_q: return

            # --- D. 绘制仪表盘 ---
            draw_dashboard(display, font_large, font_small, ego_vehicle, 'auto', None, step_idx)
            pygame.display.flip()

    finally:
        # 清理 (保留您的)
        settings = world.get_settings(); settings.synchronous_mode = False; world.apply_settings(settings)
        for actor in actor_list: actor.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()