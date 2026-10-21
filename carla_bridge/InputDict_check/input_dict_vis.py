import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from pyquaternion import Quaternion
import os

# ================= 配置区 =================
root = rf'carla_bridge/InputDict_check/input_dictC'
PKL_PATH = root + r"/step_0005.pkl"
# PKL_PATH = r"carla_bridge/InputDict_check/input_dictC/step_0005.pkl"
# OUTPUT_DIR = "carla_bridge/InputDict_check/input_dictC/vis"
OUTPUT_DIR = root + r"/vis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 导航指令映射
CMD_MAPPING = {0: 'Turn Right', 1: 'Turn Left', 2: 'Go Straight'}
# ==========================================

def get_matrix(translation, rotation):
    """ 4x4 Matrix """
    matrix = np.eye(4)
    if isinstance(rotation, (list, np.ndarray)) and len(rotation) == 4:
        matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
        matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix

def project_global_to_img(point_global, input_dict, cam_idx):
    """
    核心流水线：Global -> Ego -> Camera -> Image
    """
    # 1. 准备数据
    ego2global_t = input_dict['ego2global_translation']
    ego2global_r = input_dict['ego2global_rotation']
    ego2global = get_matrix(ego2global_t, ego2global_r)
    global2ego = np.linalg.inv(ego2global)

    camera2ego = input_dict['camera2ego'][cam_idx]
    ego2camera = np.linalg.inv(camera2ego)
    cam_intrinsic = input_dict['camera_intrinsics'][cam_idx]

    # 2. 变换计算
    point_global_hom = np.append(point_global, 1.0)
    
    # Global -> Ego -> Camera
    point_ego = global2ego @ point_global_hom
    point_cam = ego2camera @ point_ego

    # 3. 深度检查 (相机坐标系下 Z>0 为前方)
    if point_cam[2] <= 0.1: 
        return None

    # 4. 透视投影
    # cam_intrinsic 可能是 3x3 或 3x4 矩阵
    if cam_intrinsic.shape[1] == 4:
        point_img_hom = cam_intrinsic @ point_cam
    else:
        point_img_hom = cam_intrinsic @ point_cam[:3]
    
    u = point_img_hom[0] / point_img_hom[2]
    v = point_img_hom[1] / point_img_hom[2]

    return (int(u), int(v))

def transform_points(points, matrix):
    """(N, 3) 点集变换"""
    if points.shape[1] == 3:
        pts_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    else:
        pts_hom = points
    pts_trans = (matrix @ pts_hom.T).T
    return pts_trans[:, :3]

def main():
    print(f"Loading {PKL_PATH}...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    img_paths = data['img_filename']
    
    # ==================================================
    # 1. 定义测试物体 (Ego坐标系: X前, Y左, Z上)
    # ==================================================
    
    # --- 物体 A: 红点 (车后 10m, 右侧 10m, 地面) ---
    pt_dot_ego = np.array([-10.0, -10.0, 0.0, 1.0])

    # --- 物体 B: 立方体 (车前 20m, 左侧 10m, 地面) ---
    cube_center = np.array([20.0, 10.0, 0.0])
    size = 2.0
    half = size / 2
    
    # 立方体8个顶点 (Ego系)
    cube_corners_ego = np.array([
        [cube_center[0] - half, cube_center[1] - half, 0.0, 1.0],
        [cube_center[0] + half, cube_center[1] - half, 0.0, 1.0],
        [cube_center[0] + half, cube_center[1] + half, 0.0, 1.0],
        [cube_center[0] - half, cube_center[1] + half, 0.0, 1.0],
        [cube_center[0] - half, cube_center[1] - half, size, 1.0],
        [cube_center[0] + half, cube_center[1] - half, size, 1.0],
        [cube_center[0] + half, cube_center[1] + half, size, 1.0],
        [cube_center[0] - half, cube_center[1] + half, size, 1.0],
    ])
    
    # 立方体底面 (BEV用)
    cube_base_ego = np.array([
        [cube_center[0]-half, cube_center[1]-half, 0],
        [cube_center[0]+half, cube_center[1]-half, 0],
        [cube_center[0]+half, cube_center[1]+half, 0],
        [cube_center[0]-half, cube_center[1]+half, 0]
    ])
    
    # 立方体连线关系
    lines = [
        (0,1), (1,2), (2,3), (3,0),  # 底面框
        (4,5), (5,6), (6,7), (7,4),  # 顶面框
        (0,4), (1,5), (2,6), (3,7)   # 4根立柱
    ]

    # ==================================================
    # 2. 转换到 Global 坐标系
    # ==================================================
    ego2global_t = data['ego2global_translation']
    ego2global_r = data['ego2global_rotation']
    ego2global_mat = get_matrix(ego2global_t, ego2global_r)
    global2ego_mat = np.linalg.inv(ego2global_mat)

    # 转换红点
    pt_dot_global = (ego2global_mat @ pt_dot_ego)[:3]

    # 转换立方体顶点
    cube_corners_global = []
    for pt in cube_corners_ego:
        pt_global = (ego2global_mat @ pt)[:3]
        cube_corners_global.append(pt_global)

    # ==================================================
    # 3. 创建大图 (2行5列, 比例35:10)
    # ==================================================
    fig = plt.figure(figsize=(35, 10), facecolor='white')
    
    # GridSpec: 2行5列，右边2列合并为BEV
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1], 
                          wspace=0.02, hspace=0.05)

    # ==================================================
    # 4. 绘制6个摄像头图像 (左3列)
    #    顺序: 1前右, 2前, 3前左, 4后左, 5后, 6后右
    # ==================================================
    
    # 固定的摄像头位置映射 (索引 -> 网格位置)
    # 上排: 前左(0,0), 前(0,1), 前右(0,2)
    # 下排: 后左(1,0), 后(1,1), 后右(1,2)
    cam_positions = [
        (0, 1), 
        (0, 2), 
        (0, 0), 
        (1, 1),  
        (1, 2), 
        (1, 0), 
    ]
    
    cam_names = ['FRONT', 'FRONT_RIGHT', 'FRONT_LEFT', 
                 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
    
    for i, img_path in enumerate(img_paths):
        if i >= 6:
            break
        
        row, col = cam_positions[i]
        ax = fig.add_subplot(gs[row, col])
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # --- A. 绘制红点 ---
            uv_dot = project_global_to_img(pt_dot_global, data, i)
            if uv_dot and 0 <= uv_dot[0] < w and 0 <= uv_dot[1] < h:
                cv2.circle(img, uv_dot, 12, (255, 0, 0), -1)
                cv2.putText(img, "RR(-10,-10)", (uv_dot[0]+15, uv_dot[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # --- B. 绘制立方体 ---
            proj_corners = []
            for pt_g in cube_corners_global:
                uv = project_global_to_img(pt_g, data, i)
                proj_corners.append(uv)

            for start_idx, end_idx in lines:
                p1 = proj_corners[start_idx]
                p2 = proj_corners[end_idx]
                if p1 is not None and p2 is not None:
                    cv2.line(img, p1, p2, (0, 255, 255), 3)

            if proj_corners[0] is not None:
                cv2.putText(img, "FL(20,10)", proj_corners[0], 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Image Not Found', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
        
        ax.set_title(cam_names[i], fontsize=11, fontweight='bold')
        ax.axis('off')

    # ==================================================
    # 5. 绘制BEV (右2列合并, X朝上, Y朝左右)
    # ==================================================
    ax_bev = fig.add_subplot(gs[:, 3:])  # 占据所有行，第4-5列
    
    # BEV中使用的点 (Ego系)
    pt_dot_bev = np.array([-10.0, -10.0])
    
    # 注意：翻转坐标使 X朝上
    # 原本: X=前(水平右), Y=左(垂直上)
    # 翻转后: X->垂直轴(上为正), Y->水平轴(左为正)
    # 即: plot_x = ego_y, plot_y = ego_x
    
    # --- A. 绘制自车 ---
    # 车长4m(X方向), 宽2m(Y方向)
    # 翻转后: 高度=4(沿plot_y), 宽度=2(沿plot_x)
    # 中心在(0,0), 左下角: plot_x=-1, plot_y=-2
    car_rect = Rectangle((-1, -2), 2, 4, linewidth=2, 
                         edgecolor='black', facecolor='mediumseagreen', 
                         alpha=0.6, label='Ego Car', zorder=5)
    ax_bev.add_patch(car_rect)
    
    # 车头箭头 (指向上方，即X正方向)
    ax_bev.arrow(0, 0, 0, 4, head_width=0.8, head_length=0.6, 
                 fc='black', ec='black', zorder=10)
    ax_bev.text(0, 5.5, "Front (X+)", ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # 添加左右标注 (翻转后：画面左边是Left，右边是Right)
    ax_bev.text(12, 0, "← Left (Y+)", ha='center', va='center', 
                fontsize=11, color='gray')
    ax_bev.text(-12, 0, "Right (Y-) →", ha='center', va='center', 
                fontsize=11, color='gray')

    # --- B. 绘制红点 (后右: X=-10, Y=-10) ---
    # 翻转: plot_x = Y = -10, plot_y = X = -10
    ax_bev.plot(pt_dot_bev[1], pt_dot_bev[0], 'ro', markersize=15, 
                label='Red Dot: Rear-Right', zorder=8)
    ax_bev.text(pt_dot_bev[1], pt_dot_bev[0]-2, 
                f"RR\nEgo({pt_dot_bev[0]:.0f}, {pt_dot_bev[1]:.0f})", 
                color='red', ha='center', fontsize=10, fontweight='bold')

    # --- C. 绘制立方体底面 (前左: center=[20,10]) ---
    # 翻转坐标
    cube_base_flipped = np.column_stack([cube_base_ego[:, 1], cube_base_ego[:, 0]])
    poly = Polygon(cube_base_flipped, closed=True, 
                   edgecolor='cyan', facecolor='cyan', 
                   alpha=0.6, linewidth=2, label='Cyan Cube: Front-Left', zorder=7)
    ax_bev.add_patch(poly)
    ax_bev.text(cube_center[1], cube_center[0]+2, 
                f"FL\nEgo({cube_center[0]:.0f}, {cube_center[1]:.0f})", 
                color='darkcyan', ha='center', fontsize=10, fontweight='bold')

    # --- D. 历史轨迹 ---
    his_trajs = data.get('ego_his_trajs', np.zeros((0, 2)))
    if len(his_trajs) > 0:
        traj_cumsum = np.cumsum(his_trajs, axis=0)
        # 轴映射: index 0 = lateral (Y), index 1 = longitudinal (X)
        traj_y_lat = traj_cumsum[:, 0]
        traj_x_lon = traj_cumsum[:, 1]
        
        his_plot_x = np.insert(traj_y_lat, 0, 0)  # Y -> plot_x
        his_plot_y = np.insert(traj_x_lon, 0, 0)  # X -> plot_y
        
        ax_bev.plot(his_plot_x, his_plot_y, 'b-o', markersize=5, 
                    linewidth=2, label='History Trajectory', zorder=6)
        if len(his_plot_x) > 1:
            ax_bev.text(his_plot_x[-1], his_plot_y[-1]-1.5, "Past", 
                        fontsize=9, color='blue', fontweight='bold', ha='center')

    # --- E. 获取并显示指令 ---
    cmd_idx = np.argmax(data.get('ego_fut_cmd', np.array([0, 0, 1])))
    cmd_str = CMD_MAPPING.get(cmd_idx, "Unknown")

    # --- F. BEV装饰 ---
    ax_bev.set_title(f"BEV View | Command: {cmd_str}", fontsize=16, fontweight='bold')
    ax_bev.set_xlabel("Ego Y (Lateral) [m]", fontsize=12)
    ax_bev.set_ylabel("Ego X (Longitudinal) [m]", fontsize=12)
    
    ax_bev.set_aspect('equal')
    ax_bev.set_xlim(20, -20)  # Y轴翻转：左边是左侧(Y+)，右边是右侧(Y-)
    ax_bev.set_ylim(-15, 30)
    
    ax_bev.grid(True, linestyle='--', alpha=0.4)
    ax_bev.axhline(0, color='gray', lw=1)
    ax_bev.axvline(0, color='gray', lw=1)
    ax_bev.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0.02, 0.98))

    # ==================================================
    # 6. 保存
    # ==================================================
    pkl_name = os.path.basename(PKL_PATH).replace('.pkl', '')
    save_path = os.path.join(OUTPUT_DIR, f"projection_{pkl_name}.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved combined visualization to: {save_path}")
    print("-" * 50)
    print("验证说明:")
    print("1. CAM_FRONT_LEFT / CAM_FRONT: 应看到青色立方体(FL)")
    print("2. CAM_BACK / CAM_BACK_RIGHT: 应看到红点(RR)")
    print("3. BEV图: X轴朝上(前进方向), Y轴水平(左正右负)")
    print("   - 红点在右下角(后右)")
    print("   - 青色方块在左上角(前左)")

if __name__ == "__main__":
    main()