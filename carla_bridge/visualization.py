"""
VAD BEV 可视化 - 改进版
添加了 Legend 和类别颜色区分
"""
import math
import random
import os, cv2, pickle, numpy as np, sys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from pyquaternion import Quaternion
from PIL import Image

SAVE_DIR = 'carla_bridge/InputDict_check/input_dictC'
BEV_SAVE_DIR = SAVE_DIR + r'/bev_vis'
os.makedirs(BEV_SAVE_DIR, exist_ok=True)

# ============================================================
# 类别定义
# ============================================================
CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# 类别颜色 (RGB, 0-1)
CLASS_COLORS = {
    0: (1.0, 0.4, 0.4),      # car - 红色
    1: (1.0, 0.6, 0.2),      # truck - 橙色
    2: (0.8, 0.5, 0.2),      # trailer - 棕色
    3: (1.0, 1.0, 0.2),      # bus - 黄色
    4: (0.6, 0.4, 0.2),      # construction_vehicle - 深棕
    5: (0.2, 1.0, 0.2),      # bicycle - 绿色
    6: (0.2, 0.8, 0.6),      # motorcycle - 青绿
    7: (0.4, 0.4, 1.0),      # pedestrian - 蓝色
    8: (1.0, 0.6, 0.0),      # traffic_cone - 橙色锥
    9: (0.6, 0.6, 0.6),      # barrier - 灰色
}

# 地图颜色
MAP_COLORS = {
    0: 'cornflowerblue',    # divider (车道分隔线)
    1: 'orange',            # crossing (人行横道)
    2: 'slategrey',         # boundary (边界)
}
MAP_NAMES = ['Divider', 'Crossing', 'Boundary']

CMD_LIST = ['Turn Right', 'Turn Left', 'Go Straight']


def color_map(values, cmap_name='winter'):
    cmap = plt.cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    return cmap(norm(values))


def draw_ego_vehicle(ax, color='mediumseagreen', linewidth=2, alpha=0.9):
    """绘制 Ego 车辆 (绿色矩形 + 前向箭头)"""
    # 车身
    ax.plot([-0.9, -0.9], [-2, 2], color=color, linewidth=linewidth, alpha=alpha)
    ax.plot([-0.9, 0.9], [2, 2], color=color, linewidth=linewidth, alpha=alpha)
    ax.plot([0.9, 0.9], [2, -2], color=color, linewidth=linewidth, alpha=alpha)
    ax.plot([0.9, -0.9], [-2, -2], color=color, linewidth=linewidth, alpha=alpha)
    # 前向箭头
    ax.arrow(0, 0, 0, 1.5, head_width=0.5, head_length=0.3, fc=color, ec=color, alpha=alpha)


def render_box(ax, center, size, yaw, color='tomato', linewidth=1.5, label=None):
    """绘制检测框"""
    x, y = center[0], center[1]
    w, l = size[0], size[1]
    
    corners = np.array([
        [-l/2, -w/2],
        [-l/2,  w/2],
        [ l/2,  w/2],
        [ l/2, -w/2],
    ])
    
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    corners = corners @ rot.T + np.array([x, y])
    
    corners = np.vstack([corners, corners[0]])
    ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=linewidth)
    
    # 前向指示线
    front_center = (corners[2] + corners[3]) / 2
    ax.plot([x, front_center[0]], [y, front_center[1]], color=color, linewidth=linewidth)


def render_fut_trajs_grad_color(ax, center, fut_trajs, linewidth=1, cmap='autumn', fut_ts=6):
    """绘制其他车辆的预测轨迹 (渐变色)"""
    x, y = center[0], center[1]
    
    traj = np.cumsum(fut_trajs[:fut_ts], axis=0) + np.array([x, y])
    traj = np.vstack([[x, y], traj])
    
    segments = np.stack([traj[:-1], traj[1:]], axis=1)
    
    plan_vecs = []
    for seg in segments:
        x_lin = np.linspace(seg[0, 0], seg[1, 0], 51)
        y_lin = np.linspace(seg[0, 1], seg[1, 1], 51)
        xy = np.stack([x_lin, y_lin], axis=1)
        xy_seg = np.stack([xy[:-1], xy[1:]], axis=1)
        plan_vecs.append(xy_seg)
    plan_vecs = np.concatenate(plan_vecs, axis=0)
    
    y_vals = np.sin(np.linspace(0.5 * np.pi, 1.5 * np.pi, len(plan_vecs) + 1))
    colors = color_map(y_vals[:-1], cmap)
    
    lc = LineCollection(plan_vecs, colors=colors, linewidths=linewidth, cmap=cmap)
    ax.add_collection(lc)


def draw_planning_trajectory(ax, traj_offsets, cmap='winter', linewidth=2):
    """绘制 Ego 规划轨迹 (蓝色渐变)"""
    traj = traj_offsets.copy()
    traj[np.abs(traj) < 0.01] = 0.0
    traj = np.cumsum(traj, axis=0)
    traj = np.vstack([[0.0, 0.0], traj])
    traj = np.stack([traj[:-1], traj[1:]], axis=1)
    
    plan_vecs = []
    for seg in traj:
        x_lin = np.linspace(seg[0, 0], seg[1, 0], 51)
        y_lin = np.linspace(seg[0, 1], seg[1, 1], 51)
        xy = np.stack([x_lin, y_lin], axis=1)
        xy_seg = np.stack([xy[:-1], xy[1:]], axis=1)
        plan_vecs.append(xy_seg)
    plan_vecs = np.concatenate(plan_vecs, axis=0)
    
    y_vals = np.sin(np.linspace(0.5 * np.pi, 1.5 * np.pi, 301))
    colors = color_map(y_vals[:-1], cmap)
    
    lc = LineCollection(plan_vecs, colors=colors, linewidths=linewidth, linestyles='solid', cmap=cmap)
    ax.add_collection(lc)


def draw_map_lines(ax, map_pts_3d, map_labels=None, map_scores=None, score_thresh=0.5):
    """绘制地图线段"""
    if map_pts_3d is None:
        return
    maps = map_pts_3d.cpu().numpy() if hasattr(map_pts_3d, 'cpu') else map_pts_3d
    scores = map_scores.cpu().numpy() if hasattr(map_scores, 'cpu') else map_scores if map_scores is not None else np.ones(len(maps))
    labels = map_labels.cpu().numpy() if hasattr(map_labels, 'cpu') else map_labels if map_labels is not None else np.zeros(len(maps), dtype=int)
    
    for i, (line, score, label) in enumerate(zip(maps, scores, labels)):
        if score < score_thresh:
            continue
        color = MAP_COLORS.get(int(label) % len(MAP_COLORS), 'gray')
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.5, alpha=0.8, zorder=-1)


def get_plan_traj_and_cmd(result, input_dict):
    """从 result 中提取规划轨迹和命令"""
    pts_bbox = result[0]['pts_bbox']
    
    plan_traj = None
    plan_cmd = 2  # 默认直行
    
    if 'ego_fut_preds' in pts_bbox:
        ego_fut = pts_bbox['ego_fut_preds']
        if hasattr(ego_fut, 'cpu'):
            ego_fut = ego_fut.cpu().numpy()
        
        if len(ego_fut.shape) == 3 and ego_fut.shape[0] == 3:
            if 'ego_fut_cmd' in pts_bbox:
                cmd_scores = pts_bbox['ego_fut_cmd']
                if hasattr(cmd_scores, 'cpu'):
                    cmd_scores = cmd_scores.cpu().numpy()
                cmd_scores = cmd_scores.flatten()
                if len(cmd_scores) >= 3:
                    plan_cmd = int(np.argmax(cmd_scores[:3]))
            elif 'ego_fut_cmd' in input_dict:
                cmd_tensor = input_dict['ego_fut_cmd']
                if hasattr(cmd_tensor, 'cpu'):
                    cmd_tensor = cmd_tensor.cpu().numpy()
                cmd_tensor = cmd_tensor.flatten()
                if len(cmd_tensor) >= 3:
                    plan_cmd = int(np.argmax(cmd_tensor[:3]))
            
            plan_traj = ego_fut[plan_cmd]
        elif len(ego_fut.shape) == 2:
            plan_traj = ego_fut
            plan_cmd = 2
    
    return plan_traj, plan_cmd


def render_bev(result, input_dict, save_path, conf_th=0.1):
    """
    渲染 BEV 图 (带 Legend)
    
    图例说明:
    - 绿色矩形: Ego 车辆
    - 彩色矩形: 检测到的物体 (颜色按类别)
    - 蓝色渐变线: Ego 规划轨迹
    - 橙色渐变线: 其他车辆预测轨迹
    - 蓝/橙/灰线段: 地图元素
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    
    pts_bbox = result[0]['pts_bbox']
    
    # 1. 绘制地图
    if 'map_pts_3d' in pts_bbox:
        map_scores = pts_bbox.get('map_scores_3d', None)
        map_labels = pts_bbox.get('map_labels_3d', None)
        draw_map_lines(ax, pts_bbox['map_pts_3d'], map_labels, map_scores, score_thresh=0.5)
    
    # 2. 绘制检测框
    detected_classes = set()
    detection_count = 0
    
    if 'boxes_3d' in pts_bbox:
        boxes = pts_bbox['boxes_3d'].tensor.cpu().numpy()
        scores = pts_bbox['scores_3d'].cpu().numpy()
        labels = pts_bbox.get('labels_3d', None)
        if labels is not None:
            labels = labels.cpu().numpy()
        else:
            labels = np.zeros(len(boxes), dtype=int)
        
        traj_preds = None
        if 'traj_preds' in pts_bbox:
            traj_preds = pts_bbox['traj_preds'].cpu().numpy()
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score < conf_th:
                continue
            
            x, y, z, w, l, h, yaw = box[:7]
            
            # 过滤太远的检测
            if abs(x) > 30 or abs(y) > 30:
                continue
            
            detection_count += 1
            detected_classes.add(int(label))
            
            # 获取类别颜色
            color = CLASS_COLORS.get(int(label), (0.5, 0.5, 0.5))
            
            render_box(ax, [x, y], [w, l], yaw, color=color, linewidth=1.5)
            
            # 显示置信度
            ax.text(x, y - l/2 - 0.5, f'{score:.2f}', fontsize=6, 
                   ha='center', va='top', color=color)
            
            # 绘制预测轨迹
            if traj_preds is not None:
                if len(traj_preds.shape) == 3:
                    fut_traj = traj_preds[i]
                elif len(traj_preds.shape) == 4:
                    fut_traj = traj_preds[i, 0]
                else:
                    fut_traj = None
                
                if fut_traj is not None:
                    render_fut_trajs_grad_color(ax, [x, y], fut_traj, linewidth=1, cmap='autumn', fut_ts=6)
    
    # 3. 绘制 Ego 车辆
    draw_ego_vehicle(ax)
    
    # 4. 绘制规划轨迹
    plan_traj, plan_cmd = get_plan_traj_and_cmd(result, input_dict)
    if plan_traj is not None:
        draw_planning_trajectory(ax, plan_traj, cmap='winter', linewidth=2)
    
    # 5. 添加坐标轴标签
    ax.set_xlabel('Lateral (m)', fontsize=10)
    ax.set_ylabel('Longitudinal (m)', fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # 6. 创建 Legend
    legend_elements = []
    
    # Ego 车辆
    legend_elements.append(Patch(facecolor='mediumseagreen', edgecolor='mediumseagreen', 
                                  label='Ego Vehicle'))
    
    # Ego 规划轨迹
    legend_elements.append(Line2D([0], [0], color='blue', linewidth=2, 
                                   label='Ego Planned Traj'))
    
    # 其他车辆轨迹
    legend_elements.append(Line2D([0], [0], color='orange', linewidth=1.5, 
                                   label='Predicted Traj'))
    
    # 检测到的类别
    for label in sorted(detected_classes):
        if label < len(CLASS_NAMES):
            color = CLASS_COLORS.get(label, (0.5, 0.5, 0.5))
            legend_elements.append(Patch(facecolor=color, edgecolor=color, 
                                         label=f'{CLASS_NAMES[label]}'))
    
    # 地图元素
    for label, name in enumerate(MAP_NAMES):
        color = MAP_COLORS.get(label, 'gray')
        legend_elements.append(Line2D([0], [0], color=color, linewidth=1.5, 
                                       label=f'Map: {name}'))
    
    # 放置 Legend
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7, 
              framealpha=0.9, ncol=1)
    
    # 7. 标题
    cmd_str = CMD_LIST[plan_cmd] if plan_cmd < len(CMD_LIST) else "Unknown"
    ax.set_title(f'BEV | Command: {cmd_str} | Detections: {detection_count} (th={conf_th})', 
                 fontsize=11, fontweight='bold')
    
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return plan_cmd


def print_detection_summary(result):
    """打印检测结果摘要"""
    pts_bbox = result[0]['pts_bbox']
    
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    
    if 'boxes_3d' in pts_bbox:
        boxes = pts_bbox['boxes_3d'].tensor.cpu().numpy()
        scores = pts_bbox['scores_3d'].cpu().numpy()
        labels = pts_bbox['labels_3d'].cpu().numpy() if 'labels_3d' in pts_bbox else np.zeros(len(boxes))
        
        print(f"Total detections: {len(boxes)}")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # 按阈值统计
        for th in [0.5, 0.3, 0.2, 0.1, 0.05]:
            count = (scores >= th).sum()
            print(f"  Detections with score >= {th}: {count}")
        
        # 按类别统计
        print("\nBy class (score >= 0.1):")
        for label in range(10):
            mask = (labels == label) & (scores >= 0.1)
            count = mask.sum()
            if count > 0:
                class_scores = scores[mask]
                print(f"  {CLASS_NAMES[label]}: {count} (max score: {class_scores.max():.3f})")
    
    if 'map_pts_3d' in pts_bbox:
        map_scores = pts_bbox.get('map_scores_3d', None)
        if map_scores is not None:
            map_scores = map_scores.cpu().numpy() if hasattr(map_scores, 'cpu') else map_scores
            print(f"\nMap elements: {len(map_scores)}")
            print(f"Map score range: [{map_scores.min():.4f}, {map_scores.max():.4f}]")
    
    print("="*60 + "\n")


def get_lidar2img_transform(input_dict, cam_idx):
    """获取 lidar2img 变换矩阵"""
    lidar2img = input_dict.get('lidar2img', None)
    if lidar2img is not None:
        if isinstance(lidar2img, (list, tuple)) and len(lidar2img) > cam_idx:
            mat = lidar2img[cam_idx]
            if hasattr(mat, 'cpu'):
                mat = mat.cpu().numpy()
            return np.array(mat)
    return None


def render_camera_with_plan(img_path, cam_name, save_path, plan_traj_lidar, lidar2img, is_front_cam=False):
    """渲染相机图像 (前视相机显示规划轨迹)"""
    if not os.path.exists(img_path):
        fig, ax = plt.subplots(1, 1, figsize=(6, 12))
        ax.text(0.5, 0.5, "File Not Found", ha='center', va='center')
        ax.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=200, pad_inches=0.0)
        plt.close()
        return
    
    data = Image.open(img_path)
    fig, ax = plt.subplots(1, 1, figsize=(6, 12))
    ax.imshow(data)
    
    if is_front_cam and plan_traj_lidar is not None and lidar2img is not None:
        try:
            traj = plan_traj_lidar.copy()
            traj[np.abs(traj) < 0.01] = 0.0
            traj = np.cumsum(traj, axis=0)
            
            if len(traj.shape) == 1:
                if traj.shape[0] % 2 == 0:
                    traj = traj.reshape(-1, 2)
                else:
                    raise ValueError(f"Cannot reshape 1D traj")
            
            plan_traj_3d = np.concatenate([
                traj[:, [0]],
                traj[:, [1]],
                -1.0 * np.ones((traj.shape[0], 1)),
                np.ones((traj.shape[0], 1)),
            ], axis=1)
            
            plan_traj_3d = np.concatenate([np.zeros((1, 4)), plan_traj_3d], axis=0)
            plan_traj_3d[0] = [0.3, 0, -1.0, 1.0]
            
            lidar2img = np.array(lidar2img)
            plan_traj_img = lidar2img @ plan_traj_3d.T
            plan_traj_img = plan_traj_img[0:2] / np.maximum(plan_traj_img[2:3], 1e-5)
            plan_traj_img = plan_traj_img.T
            
            img_w, img_h = data.size
            valid = (plan_traj_img[:, 0] >= 0) & (plan_traj_img[:, 0] < img_w) & \
                    (plan_traj_img[:, 1] >= 0) & (plan_traj_img[:, 1] < img_h)
            
            if np.sum(valid) >= 2:
                segments = np.stack([plan_traj_img[:-1], plan_traj_img[1:]], axis=1)
                plan_vecs = []
                for seg in segments:
                    if seg.shape == (2, 2):
                        x_lin = np.linspace(seg[0, 0], seg[1, 0], 51)
                        y_lin = np.linspace(seg[0, 1], seg[1, 1], 51)
                        xy = np.stack([x_lin, y_lin], axis=1)
                        xy_seg = np.stack([xy[:-1], xy[1:]], axis=1)
                        plan_vecs.append(xy_seg)
                
                if plan_vecs:
                    plan_vecs = np.concatenate(plan_vecs, axis=0)
                    y_vals = np.sin(np.linspace(0.5 * np.pi, 1.5 * np.pi, len(plan_vecs) + 1))
                    colors = color_map(y_vals[:-1], 'winter')
                    lc = LineCollection(plan_vecs, colors=colors, linewidths=2)
                    ax.add_collection(lc)
        except Exception as e:
            print(f"Warning: Failed to render trajectory: {e}")
    
    ax.set_xlim(0, data.size[0])
    ax.set_ylim(data.size[1], 0)
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=200, pad_inches=0.0)
    plt.close()


def visualize_sample(result, input_dict, step_idx, save_dir=BEV_SAVE_DIR, conf_th=0.1):
    """完整可视化"""
    temp_dir = os.path.join(save_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 打印检测摘要
    print_detection_summary(result)
    
    # BEV
    bev_path = os.path.join(temp_dir, 'bev_pred.png')
    plan_cmd = render_bev(result, input_dict, bev_path, conf_th=conf_th)
    
    plan_traj, _ = get_plan_traj_and_cmd(result, input_dict)
    img_paths = input_dict['img_filename']
    
    cam_order_top = [2, 0, 1]
    cam_order_bot = [4, 3, 5]
    cam_name_map = {
        0: 'CAM_FRONT', 1: 'CAM_FRONT_RIGHT', 2: 'CAM_FRONT_LEFT',
        3: 'CAM_BACK', 4: 'CAM_BACK_LEFT', 5: 'CAM_BACK_RIGHT'
    }
    
    cam_imgs = {}
    for idx in range(6):
        if idx < len(img_paths):
            cam_save_path = os.path.join(temp_dir, f'cam_{idx}.png')
            is_front = (idx == 0)
            lidar2img = get_lidar2img_transform(input_dict, idx) if is_front else None
            
            render_camera_with_plan(
                img_paths[idx], cam_name_map[idx], cam_save_path,
                plan_traj if is_front else None, lidar2img, is_front
            )
            
            cam_img = cv2.imread(cam_save_path)
            cv2.putText(cam_img, cam_name_map[idx], (10, 30), 0, 1, 
                       (255, 255, 255), 3, cv2.LINE_AA)
            cam_imgs[idx] = cam_img
    
    pred_img = cv2.imread(bev_path)
    pred_img = cv2.copyMakeBorder(pred_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    
    default_cam_img = np.zeros((400, 600, 3), dtype=np.uint8)
    cam_top = cv2.hconcat([cam_imgs.get(i, default_cam_img) for i in cam_order_top])
    cam_bot = cv2.hconcat([cam_imgs.get(i, default_cam_img) for i in cam_order_bot])
    cam_img = cv2.vconcat([cam_top, cam_bot])
    
    cam_img = cv2.resize(cam_img, (2133, 800))
    pred_img = cv2.resize(pred_img, (800, 800))
    
    vis_img = cv2.hconcat([cam_img, pred_img])
    
    final_path = os.path.join(save_dir, f'vis_step_{step_idx:04d}.png')
    cv2.imwrite(final_path, vis_img)
    print(f'Saved: {final_path}')
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return final_path


def create_video_from_images(save_dir, output_path, fps=10):
    """从图像创建视频"""
    import glob
    img_files = sorted(glob.glob(os.path.join(save_dir, 'vis_step_*.png')))
    if not img_files:
        print("No images found!")
        return
    
    first_img = cv2.imread(img_files[0])
    h, w = first_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)
    
    for img_file in img_files:
        video.write(cv2.imread(img_file))
    
    video.release()
    print(f"Video saved to: {output_path}")


def main():
    sys.path.append('.')
    import projects.mmdet3d_plugin
    from vad_carla_bridge import VADAgent
    
    agent = VADAgent(
        # 'projects/configs/VAD/VAD_tiny_e2e.py',
        # 'ckpts/VAD_tiny.pth'
        'projects/configs/VAD/VAD_base_e2e.py',
        'ckpts/VAD_base.pth'
    )
    
    step_idx = 180    
    while True:
        pkl_path = os.path.join(SAVE_DIR, f'step_{step_idx:04d}.pkl')
        if not os.path.exists(pkl_path):
            print(f'No more data at step {step_idx}')
            break
        
        with open(pkl_path, 'rb') as f:
            input_dict = pickle.load(f)
        
        result = agent._forward(input_dict)
        visualize_sample(result, input_dict, step_idx, conf_th=0.2)
        
        step_idx += 1
    
    video_path = os.path.join(BEV_SAVE_DIR, 'vis.mp4')
    create_video_from_images(BEV_SAVE_DIR, video_path, fps=10)


if __name__ == '__main__':
    main()