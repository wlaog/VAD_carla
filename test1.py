from nuscenes.nuscenes import NuScenes

# 1. 初始化（确保路径正确）
nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=True)

# 2. 随便找一个样本
my_sample = nusc.sample[100]

# 3. 遍历 6 个相机打印参数
camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

for cam_name in camera_names:
    sample_data_token = my_sample['data'][cam_name]
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    
    print(f"--- {cam_name} ---")
    print(f"内参 (Intrinsic): {cs_record['camera_intrinsic']}")
    print(f"外参平移 (Translation): {cs_record['translation']}")
    print(f"外参旋转 (Rotation): {cs_record['rotation']}")