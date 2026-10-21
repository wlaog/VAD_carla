import json
import os
import cv2
import numpy as np

# --- 配置 ---
DATA_DIR = 'carla_bridge/step_debug_info'
IMG_DIR = os.path.join(DATA_DIR, 'images')
OUT_DIR = 'carla_bridge/verification_results'
if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

def verify_step(step_idx):
    # 1. 读取 JSON
    json_path = os.path.join(DATA_DIR, f'step_{step_idx:04d}.json')
    if not os.path.exists(json_path):
        print("找不到数据！")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2. 定义 3D 检查点 (车头正前方 10 米，地面)
    # 坐标系：X前, Y左, Z上 (LiDAR/Ego系)
    point_3d = np.array([10.0, 10, 0.0, 1.0]) 

    canvas = []
    cam_names = ['Front', 'Front_Right', 'Front_Left', 'Back', 'Back_Left', 'Back_Right']

    # 3. 遍历 6 张图进行投影
    for i in range(6):
        img_path = os.path.join(IMG_DIR, f'step_{step_idx:04d}', f'cam_{i}.jpg')
        img = cv2.imread(img_path)
        
        # 拿到矩阵
        l2i = np.array(data['lidar2img'][i])
        
        # 投影计算
        p_2d = l2i @ point_3d
        
        if p_2d[2] > 0.1: # 如果在相机前方
            u = int(p_2d[0] / p_2d[2])
            v = int(p_2d[1] / p_2d[2])
            
            # 画一个显眼的红点
            cv2.circle(img, (u, v), 15, (0, 0, 255), -1)
        cv2.putText(img, cam_names[i], (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        canvas.append(img)

    # 4. 拼成 2x3 的大图查看
    row1 = np.hstack(canvas[:3])
    row2 = np.hstack(canvas[3:])
    final = np.vstack([row1, row2])
    
    out_path = os.path.join(OUT_DIR, f'verify_step_{step_idx:04d}.jpg')
    cv2.imwrite(out_path, final)
    print(f"🎯 验证完成！请查看: {out_path}")

if __name__ == '__main__':
    # 您可以指定检查哪一步
    verify_step(step_idx=30)