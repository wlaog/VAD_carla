"""
对比 nuScenes 和 CARLA 的 input_dict 关键字段
"""
import pickle
import numpy as np
import os

# 路径配置
NUSCENES_DIR = 'carla_bridge/InputDict_check/input_dictN'
CARLA_DIR = 'carla_bridge/InputDict_check/input_dictC'

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compare_array(name, arr1, arr2, label1="nuScenes", label2="CARLA"):
    """对比两个数组"""
    print(f"\n{'='*60}")
    print(f"=== {name} ===")
    print(f"{'='*60}")
    
    if arr1 is None and arr2 is None:
        print("Both are None")
        return
    
    if arr1 is None:
        print(f"{label1}: None")
        print(f"{label2}: {arr2}")
        return
    
    if arr2 is None:
        print(f"{label1}: {arr1}")
        print(f"{label2}: None")
        return
    
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    
    print(f"{label1} shape: {arr1.shape}, dtype: {arr1.dtype}")
    print(f"{label2} shape: {arr2.shape}, dtype: {arr2.dtype}")
    
    if arr1.shape != arr2.shape:
        print(f"⚠️  Shape mismatch!")
    
    print(f"\n{label1}:\n{arr1}")
    print(f"\n{label2}:\n{arr2}")
    
    if arr1.shape == arr2.shape:
        diff = np.abs(arr1 - arr2)
        print(f"\nMax diff: {np.max(diff):.6f}, Mean diff: {np.mean(diff):.6f}")

def compare_lidar2img(l2i_1, l2i_2, label1="nuScenes", label2="CARLA"):
    """对比 lidar2img 列表"""
    print(f"\n{'='*60}")
    print(f"=== lidar2img (CAM_FRONT only) ===")
    print(f"{'='*60}")
    
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # 只看 CAM_FRONT
    for i in [0]:  # 只看第一个相机
        print(f"\n--- {cam_names[i]} ---")
        m1 = np.array(l2i_1[i])
        m2 = np.array(l2i_2[i])
        
        print(f"{label1}:\n{m1}")
        print(f"\n{label2}:\n{m2}")
        
        # 分析第三行（深度行）
        print(f"\n深度行分析:")
        print(f"  {label1} row2: {m1[2, :]}")
        print(f"  {label2} row2: {m2[2, :]}")
        
        # 测试投影
        pt_front = np.array([10, 0, 0, 1])  # 车前10米
        pt_left = np.array([0, 10, 0, 1])   # 车左10米
        
        r1_front = m1 @ pt_front
        r2_front = m2 @ pt_front
        r1_left = m1 @ pt_left
        r2_left = m2 @ pt_left
        
        print(f"\n投影测试 [10,0,0,1] (车前10米):")
        print(f"  {label1}: depth={r1_front[2]:.3f}")
        print(f"  {label2}: depth={r2_front[2]:.3f}")
        
        print(f"\n投影测试 [0,10,0,1] (车左10米):")
        print(f"  {label1}: depth={r1_left[2]:.3f}")
        print(f"  {label2}: depth={r2_left[2]:.3f}")

def main():
    # 加载数据
    step = 60  # 使用第60帧
    
    nusc_path = os.path.join(NUSCENES_DIR, f'step_{step:04d}.pkl')
    carla_path = os.path.join(CARLA_DIR, f'step_{step:04d}.pkl')
    
    print(f"Loading nuScenes: {nusc_path}")
    print(f"Loading CARLA: {carla_path}")
    
    if not os.path.exists(nusc_path):
        print(f"ERROR: {nusc_path} not found!")
        return
    if not os.path.exists(carla_path):
        print(f"ERROR: {carla_path} not found!")
        return
    
    nusc_data = load_pkl(nusc_path)
    carla_data = load_pkl(carla_path)
    
    print("\n" + "="*60)
    print("Available keys in nuScenes:")
    print(sorted(nusc_data.keys()))
    print("\nAvailable keys in CARLA:")
    print(sorted(carla_data.keys()))
    
    # 1. can_bus 对比
    compare_array('can_bus', 
                  nusc_data.get('can_bus'), 
                  carla_data.get('can_bus'))
    
    # 2. ego_his_trajs 对比
    compare_array('ego_his_trajs', 
                  nusc_data.get('ego_his_trajs'), 
                  carla_data.get('ego_his_trajs'))
    
    # 3. ego_fut_cmd 对比
    compare_array('ego_fut_cmd', 
                  nusc_data.get('ego_fut_cmd'), 
                  carla_data.get('ego_fut_cmd'))
    
    # 4. ego_lcf_feat 对比
    compare_array('ego_lcf_feat', 
                  nusc_data.get('ego_lcf_feat'), 
                  carla_data.get('ego_lcf_feat'))
    
    # 5. lidar2img 对比
    if 'lidar2img' in nusc_data and 'lidar2img' in carla_data:
        compare_lidar2img(nusc_data['lidar2img'], carla_data['lidar2img'])
    
    # 6. camera2ego 对比 (只看 CAM_FRONT)
    print(f"\n{'='*60}")
    print("=== camera2ego (CAM_FRONT) ===")
    print(f"{'='*60}")
    if 'camera2ego' in nusc_data and 'camera2ego' in carla_data:
        print(f"nuScenes:\n{np.array(nusc_data['camera2ego'][0])}")
        print(f"\nCARLA:\n{np.array(carla_data['camera2ego'][0])}")
    
    # 7. ego2global 对比
    compare_array('ego2global_translation',
                  nusc_data.get('ego2global_translation'),
                  carla_data.get('ego2global_translation'))
    
    compare_array('ego2global_rotation',
                  nusc_data.get('ego2global_rotation'),
                  carla_data.get('ego2global_rotation'))
    
    # 8. 图像路径检查
    print(f"\n{'='*60}")
    print("=== img_filename ===")
    print(f"{'='*60}")
    print(f"nuScenes: {nusc_data.get('img_filename', 'N/A')}")
    print(f"CARLA: {carla_data.get('img_filename', 'N/A')}")
    
    # 9. 检查图像尺寸（如果有的话）
    print(f"\n{'='*60}")
    print("=== Image size check ===")
    print(f"{'='*60}")
    try:
        import cv2
        nusc_img = cv2.imread(nusc_data['img_filename'][0])
        carla_img = cv2.imread(carla_data['img_filename'][0])
        if nusc_img is not None:
            print(f"nuScenes image shape: {nusc_img.shape}")
        else:
            print("nuScenes image: Could not load")
        if carla_img is not None:
            print(f"CARLA image shape: {carla_img.shape}")
        else:
            print("CARLA image: Could not load")
    except Exception as e:
        print(f"Could not check image sizes: {e}")

if __name__ == "__main__":
    main()