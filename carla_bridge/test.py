import pickle
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

# ================= 配置区 =================
PKL_PATH = r"carla_bridge/log/step_0018.pkl"  # 请确保此文件存在
# ==========================================

# VAD / nuScenes 默认相机顺序
CAM_NAMES = [
    "CAM_FRONT", 
    "CAM_FRONT_RIGHT", 
    "CAM_FRONT_LEFT", 
    "CAM_BACK", 
    "CAM_BACK_LEFT", 
    "CAM_BACK_RIGHT"
]

def rotation_matrix_to_euler(rot_mat):
    """将旋转矩阵转换为欧拉角 (Roll, Pitch, Yaw) - 仅供参考"""
    r = R.from_matrix(rot_mat)
    # 使用 'xyz' 顺序 (外旋)，返回角度
    return r.as_euler('xyz', degrees=True)

def main():
    print(f"Loading {PKL_PATH}...")
    try:
        with open(PKL_PATH, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("❌ Error: File not found!")
        return

    # 1. 获取关键数据列表
    # 注意：这些列表的长度通常是 6 (对应6个相机)
    intrinsics = data['camera_intrinsics'] # [4x4 matrix, ...]
    c2e_list = data['camera2ego']          # [4x4 matrix, ...]
    
    print("=" * 60)
    print(f"{'Camera Name':<20} | {'Intrinsics (fx, fy, cx, cy)':<30} | {'Extrinsics (x, y, z) [m]':<25}")
    print("=" * 60)

    for i, name in enumerate(CAM_NAMES):
        if i >= len(intrinsics): break
        
        # --- A. 解析内参 (Intrinsics) ---
        K = intrinsics[i] # 4x4 matrix
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        intr_str = f"f:({fx:.1f}, {fy:.1f}) c:({cx:.1f}, {cy:.1f})"

        # --- B. 解析外参 (Camera -> Ego) ---
        # 这里的 c2e 是将点从相机坐标系转到 Ego 坐标系的矩阵
        # Translation (最后一列) 代表相机在 Ego 系下的位置
        c2e = c2e_list[i]
        
        # 位置 (x, y, z)
        pos = c2e[:3, 3]
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        
        # 旋转 (3x3)
        rot_mat = c2e[:3, :3]
        # 转欧拉角方便看 (Roll, Pitch, Yaw)
        r, p, y = rotation_matrix_to_euler(rot_mat)
        
        print(f"{name:<20} | {intr_str:<30} | Pos: {pos_str}")
        print(f"{'':<20} | {'':<30} | Rot(deg): R={r:.1f}, P={p:.1f}, Y={y:.1f}")
        print("-" * 60)

    print("\n💡 提示: ")
    print("1. Intrinsics: fx/fy 应该接近，cx/cy 应该在图像中心 (例如 800/2=400)。")
    print("2. Extrinsics Pos: 这是相机相对于车身中心(后轴)的物理位置。")
    print("   - CAM_FRONT x 应该是正数 (车头方向)")
    print("   - CAM_BACK x 应该是负数 (车尾方向)")
    print("   - CAM_BACK_LEFT y 应该是正数 (左侧)")

if __name__ == "__main__":
    main()