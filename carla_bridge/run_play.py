import math
import random
import os, cv2, pickle, numpy as np, sys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec  # 引入网格布局模块

SAVE_DIR = 'carla_bridge/InputDict_check/input_dictC'
BEV_SAVE_DIR = 'carla_bridge/InputDict_check/input_dictC/bev_vis'
os.makedirs(BEV_SAVE_DIR, exist_ok=True)

# 地图类别颜色
MAP_COLORS = ['cornflowerblue', 'royalblue', 'slategrey']
# 指令列表 (对应索引 0, 1, 2)
CMD_LIST = ['Turn Right', 'Turn Left', 'Go Straight']

def color_map(values, cmap_name='winter'):
    """生成渐变颜色"""
    cmap = plt.cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    return cmap(norm(values))

def draw_ego_vehicle(ax, color='mediumseagreen', linewidth=1.5):
    """绘制自车矩形框"""
    ax.plot([-0.9, -0.9], [-2, 2], color=color, linewidth=linewidth)
    ax.plot([-0.9, 0.9], [2, 2], color=color, linewidth=linewidth)
    ax.plot([0.9, 0.9], [2, -2], color=color, linewidth=linewidth)
    ax.plot([0.9, -0.9], [-2, -2], color=color, linewidth=linewidth)
    ax.plot([0.0, 0.0], [0.0, 2], color=color, linewidth=linewidth)

def draw_planning_trajectory(ax, traj_offsets, cmap='winter'):
    """绘制规划轨迹（带渐变色），使用 cumsum"""
    traj = traj_offsets.copy()
    traj[np.abs(traj) < 0.01] = 0.0
    traj = np.cumsum(traj, axis=0) # 累积求和
    traj = np.vstack([[0.0, 0.0], traj]) # 添加起点
    
    segments = np.stack([traj[:-1], traj[1:]], axis=1)
    y = np.sin(np.linspace(0.5 * np.pi, 1.5 * np.pi, len(segments) * 50 + 1))
    colors = color_map(y[:-1], cmap)
    
    plan_vecs = []
    for seg in segments:
        x_lin = np.linspace(seg[0, 0], seg[1, 0], 51)
        y_lin = np.linspace(seg[0, 1], seg[1, 1], 51)
        xy = np.stack([x_lin, y_lin], axis=1)
        xy_seg = np.stack([xy[:-1], xy[1:]], axis=1)
        plan_vecs.append(xy_seg)
    plan_vecs = np.concatenate(plan_vecs, axis=0)
    
    line_collection = LineCollection(plan_vecs, colors=colors, linewidths=2)
    ax.add_collection(line_collection)
    return traj

def draw_obstacle_box(ax, box, score, score_thresh=0.3, color='tomato'):
    """绘制障碍物边界框"""
    if score < score_thresh: return
    x, y, w, l, yaw = box[0], box[1], box[3], box[4], box[6]
    if abs(x) > 15 or abs(y) > 30: return
    
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    corners = np.array([[-l/2, -w/2], [-l/2, w/2], [l/2, w/2], [l/2, -w/2], [-l/2, -w/2]])
    rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    corners_rot = corners @ rot_matrix.T + np.array([x, y])
    ax.plot(corners_rot[:, 0], corners_rot[:, 1], color=color, linewidth=1)

def draw_obstacle_trajectory(ax, box, fut_trajs, score, score_thresh=0.3, cmap='autumn'):
    """绘制障碍物预测轨迹"""
    if score < score_thresh: return
    x, y = box[0], box[1]
    if abs(x) > 15 or abs(y) > 30: return
    
    traj = np.cumsum(fut_trajs, axis=0) + np.array([x, y])
    traj = np.vstack([[x, y], traj])
    
    segments = np.stack([traj[:-1], traj[1:]], axis=1)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(segments)))
    for seg, c in zip(segments, colors):
        ax.plot(seg[:, 0], seg[:, 1], color=c, linewidth=1, alpha=0.8)

def draw_map_pts_direct(ax, map_pts_3d, map_scores=None, score_thresh=0.5):
    """绘制地图点"""
    if map_pts_3d is None: return
    maps = map_pts_3d.cpu().numpy() if hasattr(map_pts_3d, 'cpu') else map_pts_3d
    scores = map_scores.cpu().numpy() if hasattr(map_scores, 'cpu') else map_scores if map_scores is not None else np.ones(len(maps))
    
    for i, (line, score) in enumerate(zip(maps, scores)):
        if score < score_thresh: continue
        color = MAP_COLORS[i % len(MAP_COLORS)]
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1, alpha=0.6, linestyle='--')

def visualize_bev(result, input_dict, step_idx, save_dir=BEV_SAVE_DIR):
    """
    完整的可视化：左侧6路相机 + 右侧BEV + 顶部状态信息
    """
    # 创建大图：宽20，高10，白色背景
    fig = plt.figure(figsize=(35, 10), facecolor='white')
    
    # 定义网格布局：2行5列
    # 左侧3列用于放图片(2x3)，右侧2列合并用于放BEV
    gs = gridspec.GridSpec(2, 5, figure=fig)
    
    # ============ 1. 绘制左侧环视图片 ============
    img_paths = input_dict['img_filename']
    
    # VAD/NuScenes数据通常顺序: [Front, Front_Right, Front_Left, Back, Back_Left, Back_Right]
    # 我们希望显示的布局 (符合直觉):
    #  [Front_Left]  [Front]   [Front_Right]
    #  [Back_Left]   [Back]    [Back_Right]
    
    # 建立 列表索引 -> Grid坐标(row, col) 的映射
    # 假设 input_dict['img_filename'] 顺序为 [F, FR, FL, B, BL, BR] (这是mmdet3d常见顺序)
    # 如果你的顺序不同，请调整这里的 map_idx
    
    # 视觉位置: (0,0)=FL, (0,1)=F, (0,2)=FR
    #          (1,0)=BL, (1,1)=B, (1,2)=BR
    
    # 对应的列表索引 (根据你之前的配置 configs 顺序推断)
    # 0:Front, 1:F_Right, 2:F_Left, 3:Back, 4:B_Left, 5:B_Right
    idx_map = {
        2: (0, 0, 'Front Left'),  0: (0, 1, 'Front'),  1: (0, 2, 'Front Right'),
        4: (1, 0, 'Back Left'),   3: (1, 1, 'Back'),   5: (1, 2, 'Back Right')
    }

    for list_idx, (row, col, title) in idx_map.items():
        if list_idx < len(img_paths):
            ax_img = fig.add_subplot(gs[row, col])
            
            # 读取图片 (matplotlib读取默认为RGB，cv2为BGR)
            # 假设路径是相对路径，如果报错请检查路径
            if os.path.exists(img_paths[list_idx]):
                # 使用 cv2 读取并转 RGB，防止颜色错误
                img_bgr = cv2.imread(img_paths[list_idx])
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax_img.imshow(img_rgb)
                else:
                    ax_img.text(0.5, 0.5, "Load Fail", ha='center')
            else:
                ax_img.text(0.5, 0.5, "File Not Found", ha='center')
            
            ax_img.set_title(title, fontsize=10, color='black')
            ax_img.axis('off')

    # ============ 2. 绘制右侧 BEV ============
    # 占用右侧的两列 (col 3, 4)，跨越两行
    ax_bev = fig.add_subplot(gs[:, 3:])
    
    # 设置 BEV 坐标范围
    ax_bev.set_xlim(-30, 30)
    ax_bev.set_ylim(-30, 30)
    
    pts_bbox = result[0]['pts_bbox']
    
    # 2.1 地图
    if 'map_pts_3d' in pts_bbox:
        map_scores = pts_bbox.get('map_scores_3d', None)
        draw_map_pts_direct(ax_bev, pts_bbox['map_pts_3d'], map_scores, score_thresh=0.5)
    
    # 2.2 障碍物
    if 'boxes_3d' in pts_bbox:
        boxes = pts_bbox['boxes_3d'].tensor.cpu().numpy()
        scores = pts_bbox['scores_3d'].cpu().numpy()
        
        traj_preds = None
        if 'traj_preds' in pts_bbox:
            traj_preds = pts_bbox['traj_preds'].cpu().numpy()
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            draw_obstacle_box(ax_bev, box, score, score_thresh=0.3)
            if traj_preds is not None and score > 0.3:
                if len(traj_preds.shape) == 3:
                    draw_obstacle_trajectory(ax_bev, box, traj_preds[i], score)
                elif len(traj_preds.shape) == 4:
                    draw_obstacle_trajectory(ax_bev, box, traj_preds[i, 0], score)
    
    # 2.3 自车
    draw_ego_vehicle(ax_bev)
    
    # 2.4 历史轨迹
    if 'ego_his_trajs' in input_dict:
        his_trajs = input_dict['ego_his_trajs']
        if hasattr(his_trajs, 'cpu'): his_trajs = his_trajs.cpu().numpy()
        # 注意：这里假设 ego_his_trajs 已经是绝对坐标或已累加
        # 如果是相对偏移且全是0，这里画出来就是一个点
        ax_bev.plot(his_trajs[:, 1], his_trajs[:, 0], 'c--', alpha=0.6, linewidth=2, label='History')
    
    # 2.5 规划轨迹
    if 'ego_fut_preds' in pts_bbox:
        ego_fut = pts_bbox['ego_fut_preds']
        if hasattr(ego_fut, 'cpu'): ego_fut = ego_fut.cpu().numpy()
        traj_offsets = ego_fut[2] if len(ego_fut.shape) == 3 else ego_fut
        draw_planning_trajectory(ax_bev, traj_offsets, cmap='winter')

    # ============ 3. 绘制指令与状态文本 ============
    # 获取指令
    cmd_idx = 2 # 默认 Go Straight
    if 'ego_fut_cmd' in input_dict:
        cmd_tensor = input_dict['ego_fut_cmd']
        if hasattr(cmd_tensor, 'cpu'): cmd_tensor = cmd_tensor.cpu().numpy()
        # 通常是 [1, 3] 或者 [1, 1, 1, 3] 的 one-hot，取 argmax
        if cmd_tensor.size > 0:
            cmd_idx = np.argmax(cmd_tensor)
            # 防止 scalar 情况
            if isinstance(cmd_idx, np.ndarray): cmd_idx = cmd_idx.item()
            
    cmd_str = CMD_LIST[cmd_idx] if cmd_idx < len(CMD_LIST) else "Unknown"
    
    # 在 BEV 图上方或角落绘制指令
    # 使用 bbox props 增加边框使其更清晰
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax_bev.text(0, 25, f'CMD: {cmd_str}', fontsize=14, fontweight='bold', 
                color='black', ha='center', bbox=props)
    
    # BEV 设置
    ax_bev.set_aspect('equal')
    ax_bev.axhline(0, color='grey', lw=0.5, alpha=0.5)
    ax_bev.axvline(0, color='grey', lw=0.5, alpha=0.5)
    ax_bev.set_xlabel('X (m)')
    ax_bev.set_ylabel('Y (m)')
    ax_bev.set_title(f'Step {step_idx:04d}', fontsize=12)
    ax_bev.grid(True, alpha=0.3)
    
    # 保存
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'bev_step_{step_idx:04d}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=100, facecolor='white')
    plt.close()
    print(f'Saved visualization: {save_path}')

def main():
    sys.path.append('.')
    import projects.mmdet3d_plugin
    from vad_carla_bridge import VADAgent
    
    # 初始化
    agent = VADAgent(
        'projects/configs/VAD/VAD_tiny_e2e.py',
        'ckpts/VAD_tiny.pth'
    )
    
    step_idx = 9
    while True:
        pkl_path = os.path.join(SAVE_DIR, f'step_{step_idx:04d}.pkl')
        if not os.path.exists(pkl_path):
            print(f'No more data at step {step_idx}')
            break
        
        with open(pkl_path, 'rb') as f:
            input_dict = pickle.load(f)
        
        # 运行模型
        result = agent._forward(input_dict)
        a = agent.run_step(input_dict)  # 仅为了打印日志
        # 可视化
        visualize_bev(result, input_dict, step_idx)
        
        step_idx += 1

if __name__ == '__main__':
    main()