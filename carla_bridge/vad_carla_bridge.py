import math
import numpy as np
import torch
import carla
import cv2
import pickle
import json
import os
import sys
from collections import deque
from pyquaternion import Quaternion
from mmcv import Config
from mmdet3d.apis import init_model
from mmdet3d.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode, get_box_type
from mmdet3d.core.points import LiDARPoints
from mmdet.datasets.pipelines import to_tensor
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from mmcv.parallel import DataContainer as DC
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
import matplotlib.pyplot as plt

import carla
import numpy as np
import math
from collections import deque
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
from mmdet3d.core.points import LiDARPoints
import torch

import carla
import numpy as np
import math
from collections import deque
from pyquaternion import Quaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
from mmdet3d.core.points import LiDARPoints
import torch

class VADAdapter:
    def __init__(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle
        self.his_buffer = deque(maxlen=20)  # 增大缓冲区以支持更长历史
        self.last_yaw = None
        self.last_timestamp = None
        
        # 图像参数
        center_x, center_y = 800.0, 450.0
        
        # 相机配置: (x, y, z, yaw, fx)
        self.cam_configs = [
            (1.72, 0.00, 1.49, 0.0, 1252.8),    # CAM_FRONT
            (1.58, 0.50, 1.52, 55.0, 1256.7),   # CAM_FRONT_RIGHT
            (1.58, -0.50, 1.51, -55.0, 1257.9), # CAM_FRONT_LEFT
            (0.06, -0.01, 1.57, 180.0, 796.9),  # CAM_BACK
            (1.05, -0.48, 1.56, -118.0, 1255.0),# CAM_BACK_LEFT
            (1.06, 0.47, 1.55, 113.0, 1250.0)   # CAM_BACK_RIGHT
        ]
        
        # 构建内参矩阵
        self.intrinsics = []
        for _, _, _, _, fx in self.cam_configs:
            fy = fx
            intrinsic = np.array([
                [fx, 0.0, center_x],
                [0.0, fy, center_y],
                [0.0, 0.0, 1.0]
            ])
            self.intrinsics.append(intrinsic)
    
    def _build_lidar2img_nuscenes_style(self):
        """
        用 nuScenes 完全相同的方式构建 lidar2img
        """
        lidar2img_list = []
        lidar2cam_list = []
        camera2ego_list = []
        cam_intrinsics_list = []
        
        swap_xy = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        for i, (x, y, z, yaw_deg, fx) in enumerate(self.cam_configs):
            cam_translation = np.array([x, -y, z])
            yaw_rad = np.radians(-yaw_deg)
            
            optical2ego_base = np.array([
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ])
            
            c, s = np.cos(yaw_rad), np.sin(yaw_rad)
            yaw_rotation = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]
            ])
            
            cam2ego_rotation = yaw_rotation @ optical2ego_base
            
            sensor2lidar_rotation = cam2ego_rotation
            sensor2lidar_translation = cam_translation
            
            lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
            lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
            
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            
            intrinsic = self.intrinsics[i]
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic
            
            lidar2img_raw = viewpad @ lidar2cam_rt.T
            lidar2img = lidar2img_raw @ swap_xy
            
            lidar2cam_raw = lidar2cam_rt.T
            lidar2cam = lidar2cam_raw @ swap_xy
            
            camera2ego = np.eye(4)
            camera2ego[:3, :3] = cam2ego_rotation
            camera2ego[:3, 3] = cam_translation
            
            lidar2img_list.append(lidar2img)
            lidar2cam_list.append(lidar2cam)
            camera2ego_list.append(camera2ego)
            cam_intrinsics_list.append(viewpad)
        
        return lidar2img_list, lidar2cam_list, camera2ego_list, cam_intrinsics_list


    def get_can_bus(self, timestamp):
        t = self.ego_vehicle.get_transform()
        v = self.ego_vehicle.get_velocity()
        a = self.ego_vehicle.get_acceleration()
        rot = t.rotation
        
        # 转换到 nuScenes 坐标系 (Y 取反)
        pos_x, pos_y, pos_z = t.location.x, -t.location.y, t.location.z
        vel_x, vel_y, vel_z = v.x, -v.y, v.z
        acc_x, acc_y, acc_z = a.x, -a.y, a.z
        
        yaw_rad = math.radians(-rot.yaw)
        yaw_rad = (yaw_rad + math.pi) % (2 * math.pi) - math.pi
        
        # 保存历史位置
        self.his_buffer.append(np.array([pos_x, pos_y, yaw_rad, timestamp]))
        
        # 计算 yaw_rate
        if self.last_yaw is not None and self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > 0:
                yaw_rate = (yaw_rad - self.last_yaw) / dt
                if yaw_rate > np.pi / dt:
                    yaw_rate -= 2 * np.pi / dt
                elif yaw_rate < -np.pi / dt:
                    yaw_rate += 2 * np.pi / dt
            else:
                yaw_rate = 0.0
        else:
            yaw_rate = 0.0
        
        self.last_yaw = yaw_rad
        self.last_timestamp = timestamp
        
        q = Quaternion(axis=[0, 0, 1], radians=yaw_rad)
        speed = math.sqrt(vel_x**2 + vel_y**2)
        
        can_bus = np.zeros(18)
        can_bus[0:3] = [pos_x, pos_y, pos_z]
        can_bus[3:7] = [q.w, q.x, q.y, q.z]
        can_bus[7:10] = [vel_x, vel_y, vel_z]
        can_bus[10:13] = [acc_x, acc_y, acc_z]
        can_bus[13] = speed
        can_bus[15] = yaw_rate
        can_bus[16] = yaw_rad
        can_bus[17] = math.degrees(yaw_rad)
        return can_bus

    def get_relative_his_trajs(self, current_loc, yaw):
        """
        获取历史轨迹 (相对于当前位置和朝向)
        
        关键修复:
        1. nuScenes 是 2Hz，每步 0.5s；CARLA 是 10Hz，每步 0.1s
           所以需要跨 5 帧采样来匹配时间步长
        2. 输出格式与 lidar2img 的 swap_xy 一致：[Y, X] 即 [横向, 纵向]
        """
        # nuScenes 时间步长是 0.5s，CARLA 是 0.1s
        # 需要跨 5 帧采样
        FRAME_SKIP = 5
        
        if len(self.his_buffer) < FRAME_SKIP * 2 + 1:
            return np.zeros((2, 2), dtype=np.float32)
        
        curr_pos = np.array([current_loc.x, -current_loc.y])
        
        # 旋转矩阵
        c, s = math.cos(-yaw), math.sin(-yaw)
        rot_mat = np.array([[c, -s], [s, c]])
        
        buffer_list = list(self.his_buffer)
        rel_trajs = []
        
        # 从最新往回，每隔 FRAME_SKIP 帧采样
        # 索引: -1 (当前), -6 (0.5s前), -11 (1.0s前)
        for i in range(2):
            curr_idx = -(1 + i * FRAME_SKIP)
            prev_idx = curr_idx - FRAME_SKIP
            
            if abs(prev_idx) <= len(buffer_list) and abs(curr_idx) <= len(buffer_list):
                curr = buffer_list[curr_idx][:2]
                prev = buffer_list[prev_idx][:2]
                diff = curr - prev  # 从过去到现在的位移
                rel_diff = rot_mat @ diff
                # 交换 X 和 Y，与 lidar2img 的 swap_xy 一致
                # nuScenes 格式是 [横向(Y), 纵向(X)]
                rel_trajs.append(np.array([rel_diff[1], rel_diff[0]]))
            else:
                rel_trajs.append(np.zeros(2))
        
        return np.array(rel_trajs, dtype=np.float32)

    def get_ego_lcf_feat(self, timestamp):
        """
        获取 ego_lcf_feat (Local Coordinate Frame features)
        
        nuScenes 格式分析:
        [0]: steering
        [1]: speed
        [2]: vel_x (纵向速度，前为正)
        [3]: vel_y (横向速度，左为正)
        [4]: yaw_rate
        [5]: delta_x (0.5s 内的纵向位移，约 4m @ 8m/s)
        [6]: delta_y (0.5s 内的横向位移)
        [7]: speed
        [8]: delta_yaw 或其他角度相关
        
        关键修复:
        1. 时间步长：delta 应该是 0.5s 的位移，不是 0.1s
        2. 坐标顺序：vel_x 是纵向(前)，vel_y 是横向(左)
        """
        v = self.ego_vehicle.get_velocity()
        a = self.ego_vehicle.get_acceleration()
        ctrl = self.ego_vehicle.get_control()
        
        # 转换到 nuScenes 坐标系
        # nuScenes: X前, Y左
        vel_x = v.x      # 纵向速度 (前为正)
        vel_y = -v.y     # 横向速度 (左为正，CARLA Y右所以取反)
        
        speed = math.sqrt(vel_x**2 + vel_y**2)
        
        # 计算 0.5s 的位移 (跨 5 帧)
        FRAME_SKIP = 5
        delta_x, delta_y = 0.0, 0.0
        delta_yaw = 0.0
        yaw_rate = 0.0
        
        if len(self.his_buffer) >= FRAME_SKIP + 1:
            curr = self.his_buffer[-1]
            prev = self.his_buffer[-(FRAME_SKIP + 1)]
            
            delta_x = curr[0] - prev[0]  # 纵向位移
            delta_y = curr[1] - prev[1]  # 横向位移
            delta_yaw = curr[2] - prev[2]
            
            # 处理角度跳变
            if delta_yaw > np.pi:
                delta_yaw -= 2 * np.pi
            elif delta_yaw < -np.pi:
                delta_yaw += 2 * np.pi
            
            dt = curr[3] - prev[3]
            if dt > 0:
                yaw_rate = delta_yaw / dt
        
        ego_lcf_feat = np.array([
            ctrl.steer,          # [0] steering
            speed,               # [1] speed  
            vel_x,               # [2] vel_x (纵向)
            vel_y,               # [3] vel_y (横向)
            yaw_rate,            # [4] yaw_rate
            delta_x,             # [5] delta_x (纵向位移，0.5s)
            delta_y,             # [6] delta_y (横向位移，0.5s)
            speed,               # [7] speed
            delta_yaw            # [8] delta_yaw
        ], dtype=np.float32)
        
        return ego_lcf_feat

    def make_input_dict(self, step_idx, timestamp, img_paths, cam_actors):
        t = self.ego_vehicle.get_transform()
        
        yaw_nusc = math.radians(-t.rotation.yaw)
        yaw_nusc = (yaw_nusc + math.pi) % (2 * math.pi) - math.pi
        q = Quaternion(axis=[0, 0, 1], radians=yaw_nusc)
        location_rhs = np.array([t.location.x, -t.location.y, t.location.z])

        ego2global = np.eye(4)
        ego2global[:3, :3] = q.rotation_matrix
        ego2global[:3, 3] = location_rhs
        lidar2global = ego2global

        lidar2img_list, lidar2cam_list, camera2ego_list, cam_intrinsics_list = \
            self._build_lidar2img_nuscenes_style()

        can_bus = self.get_can_bus(timestamp)
        ego_his_trajs = self.get_relative_his_trajs(t.location, yaw_nusc)
        ego_lcf_feat = self.get_ego_lcf_feat(timestamp)

        input_dict = {
            'sample_idx': f"STEP_{step_idx:04d}",
            'pts_filename': "dummy_path",
            'sweeps': [],
            'scene_token': "carla_scene_001",
            'frame_idx': step_idx,
            'timestamp': timestamp,
            'img_filename': img_paths,
            'map_location': "singapore-onenorth",
            'fut_valid_flag': True,
            'can_bus': can_bus,
            'ego_his_trajs': ego_his_trajs,
            'ego_lcf_feat': ego_lcf_feat,
            
            'lidar2img': lidar2img_list,
            'camera_intrinsics': cam_intrinsics_list,
            'cam_intrinsic': cam_intrinsics_list,
            'lidar2cam': lidar2cam_list,
            'camera2ego': camera2ego_list,
            
            'lidar2global': lidar2global,
            'ego2global_translation': location_rhs,
            'ego2global_rotation': [q.w, q.x, q.y, q.z],
            'lidar2ego': np.eye(4),
            'lidar2ego_translation': [0, 0, 0],
            'lidar2ego_rotation': [1, 0, 0, 0],
            
            'prev_idx': None, 'next_idx': None,
            'ego_fut_trajs': np.zeros((6, 2)), 
            'ego_fut_masks': np.ones(6), 
            'ego_fut_cmd': np.array([0, 0, 1]), 
            'img_fields': [], 'bbox3d_fields': [], 'pts_mask_fields': [], 
            'pts_seg_fields': [], 'bbox_fields': [], 'mask_fields': [], 'seg_fields': [],
            'box_type_3d': LiDARInstance3DBoxes, 'box_mode_3d': Box3DMode.LIDAR,
            'points': LiDARPoints(torch.zeros((0, 5)), points_dim=5),
            'gt_bboxes_3d': LiDARInstance3DBoxes(torch.zeros((0, 9)), box_dim=9),
            'gt_labels_3d': np.array([], dtype=np.int64),
            'attr_labels': np.zeros((0, 20), dtype=np.float32), 
            'ann_info': {'gt_bboxes_3d': [], 'gt_labels_3d': []}
        }
        return input_dict

class VADAgent():
    def __init__(self, config_path, checkpoint_path, device='cuda:0',
        *args,
        **kwargs):
        
        map_fixed_ptsnum_per_line=-1,
        box_type_3d='LiDAR',
        # self.vector_map = VectorizedLocalMap(kwargs['data_root'], 
        #                     patch_size=self.patch_size, map_classes=self.MAPCLASSES, 
        #                     fixed_ptsnum_per_line=map_fixed_ptsnum_per_line,
        #                     padding_value=self.padding_value)
        self.cfg = Config.fromfile(config_path)
        self.model = init_model(config_path, checkpoint_path, device=device); self.model.eval()
        p_cfg = self.cfg.data.test.pipeline 
        p_cfg = [s for s in self.cfg.data.test.pipeline if s['type'] not in ['LoadPointsFromFile', 'LoadAnnotations3D']]
        self.pipeline = Compose(p_cfg)
        # self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        # results['box_type_3d'] = self.box_type_3d
        # results['box_mode_3d'] = self.box_mode_3d
        results['box_type_3d'] = LiDARInstance3DBoxes
        results['box_mode_3d'] = Box3DMode.LIDAR
        

    def run_step(self, input_dict):
        # self.debug_images(input_dict)
        result = self._forward(input_dict)

        # pts_bbox = result[0]['pts_bbox']

        # print("=== Detection Results ===")
        # print(f"boxes_3d shape: {pts_bbox['boxes_3d'].tensor.shape if 'boxes_3d' in pts_bbox else 'N/A'}")
        # print(f"scores_3d: {pts_bbox['scores_3d'] if 'scores_3d' in pts_bbox else 'N/A'}")
        # print(f"labels_3d: {pts_bbox['labels_3d'] if 'labels_3d' in pts_bbox else 'N/A'}")

        # # 检查是否有任何检测结果（即使置信度低）
        # if 'scores_3d' in pts_bbox:
        #     scores = pts_bbox['scores_3d'].cpu().numpy()
        #     print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        #     print(f"Detections with score > 0.1: {(scores > 0.1).sum()}")
        #     print(f"Detections with score > 0.01: {(scores > 0.01).sum()}")


        fut_cmd = np.argmax(result[0]['pts_bbox']['ego_fut_cmd'].cpu().numpy())
        plan_traj = result[0]['pts_bbox']['ego_fut_preds'][fut_cmd].cpu().numpy()
        plan_traj[abs(plan_traj) < 0.01] = 0.0
        plan_traj = plan_traj.cumsum(axis=0)
        return plan_traj
    
    
    def _forward(self, input_dict):
        self.pre_pipeline(input_dict)
        data = self.pipeline(input_dict)
        # data = self.vectormap_pipeline(data, input_dict)
        data = scatter(collate([data], samples_per_gpu=1), [0])[0]
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        return result

    # 检查输入图像

    def debug_images(self, input_dict):
        img_paths = input_dict['img_filename']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        cam_names = ['FRONT', 'FRONT_RIGHT', 'FRONT_LEFT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
        
        for i, (path, ax) in enumerate(zip(img_paths, axes.flat)):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(cam_names[i])
            ax.axis('off')
        
        plt.savefig('debug_input_images.png')
        print("Saved debug_input_images.png")

import carla
import numpy as np
import math
from collections import deque

class PIDController:
    def __init__(self, wheelbase=2.5, dt=0.1):
        self.dt = dt
        self.wheelbase = wheelbase  # Audi TT 轴距约 2.5 米
        
        # --- 纵向控制 (速度 PID) ---
        self.lon_kp = 1.0
        self.lon_ki = 0.15
        self.lon_kd = 0.05
        self.lon_error_buffer = deque(maxlen=10)
        self.lon_integral = 0.0
        
        # --- 横向控制系数 ---
        # 1.0 弧度约 57 度，CARLA 最大转角约 70 度，因此系数 0.8 较为适中
        self.steer_gain = 0.8 

    def _pid_step(self, error, buffer, integral, kp, ki, kd):
        buffer.append(error)
        p = kp * error
        integral += error * self.dt
        integral = np.clip(integral, -2.0, 2.0) # 抗饱和
        i = ki * integral
        d = kd * (buffer[-1] - buffer[-2]) / self.dt if len(buffer) >= 2 else 0.0
        return p + i + d, integral

    def get_target_speed(self, traj):
        """
        从轨迹推算期望速度 (m/s)
        """
        if len(traj) < 2: return 0.0
        # 使用第 4 个点 (2.0s 处) 计算目标车速更稳定
        target_idx = min(3, len(traj)-1)
        point = traj[target_idx]
        dist = math.sqrt(point[0]**2 + point[1]**2)
        time = (target_idx + 1) * 0.5 # 每个点 0.5s
        return dist / time

    def get_lookahead_point(self, traj, current_speed):
        """
        动态预瞄：车速快看远，车速慢看近
        """
        lookahead_dist = 3.0 + 0.5 * current_speed
        best_point = traj[0]
        min_diff = float('inf')
        for point in traj:
            dist = math.sqrt(point[0]**2 + point[1]**2)
            diff = abs(dist - lookahead_dist)
            if diff < min_diff:
                min_diff = diff
                best_point = point
            if dist > lookahead_dist + 5.0: break
        return best_point, lookahead_dist

    def get_control(self, traj, current_speed):
        """
        Args:
            traj: [x(前), y(左)]
            current_speed: 必须是 m/s
        """
        # 1. 纵向 PID
        target_speed = self.get_target_speed(traj)
        speed_error = target_speed - current_speed
        acc_cmd, self.lon_integral = self._pid_step(
            speed_error, self.lon_error_buffer, self.lon_integral,
            self.lon_kp, self.lon_ki, self.lon_kd
        )
        
        # 2. 横向纯追踪
        target_point, ld = self.get_lookahead_point(traj, current_speed)
        tx, ty = target_point[0], target_point[1] # tx 前, ty 左
        
        # 计算 alpha 角 (车头方向与目标点连线的夹角)
        alpha = math.atan2(ty, tx)
        
        # 纯追踪公式: delta = atan(2 * L * sin(alpha) / Ld)
        steer_rad = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        
        # 映射到 CARLA [-1, 1]。注意：左转(正角)对应 CARLA 的负 steer
        steer_cmd = -1.0 * self.steer_gain * steer_rad
        
        control = carla.VehicleControl()
        control.steer = np.clip(steer_cmd, -1.0, 1.0)
        
        if acc_cmd >= 0:
            control.throttle = np.clip(acc_cmd, 0.0, 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = np.clip(-acc_cmd, 0.0, 1.0)
            
        if target_speed < 0.1: # 目标速度太低则刹停
            control.throttle = 0.0; control.brake = 1.0
            
        return control