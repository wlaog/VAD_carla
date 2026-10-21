import os, sys, torch, pickle, numpy as np
from mmcv import Config
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
from mmdet3d.core.points import LiDARPoints
from mmdet3d.apis import init_model
from mmcv.parallel import collate, scatter

# --- 终极对齐类：补全 Agent 特征维度 ---
class VADCARLADataset(NuScenesDataset):
    def __init__(self, pkl_path, pipeline_cfg):
        self.pkl_path = pkl_path
        self.test_mode = True
        self.box_type_3d = LiDARInstance3DBoxes
        self.box_mode_3d = Box3DMode.LIDAR
        
        from mmdet3d.datasets.pipelines import Compose
        self.pipeline = Compose(pipeline_cfg)

    def get_data_info(self, index):
        with open(self.pkl_path, 'rb') as f:
            input_dict = pickle.load(f)

        # 补全基础元数据
        input_dict.update({k: [] for k in ['img_fields', 'bbox3d_fields', 'pts_mask_fields', 
                                          'pts_seg_fields', 'bbox_fields', 'mask_fields', 'seg_fields']})
        input_dict['box_type_3d'] = self.box_type_3d
        input_dict['box_mode_3d'] = self.box_mode_3d
        input_dict['points'] = LiDARPoints(torch.zeros((0, 5)), points_dim=5)
        
        # 补全平铺标注信息
        input_dict.update(self.get_ann_info(index))
        return input_dict

    def get_ann_info(self, index):
        """
        核心修复：VAD 要求属性标签必须能 reshape 成 (6, 2)
        我们提供 20 维全 0 向量 (12位轨迹 + 8位其他特征)，确保 reshape 不报错
        """
        return {
            'gt_bboxes_3d': LiDARInstance3DBoxes(torch.zeros((1, 9)), box_dim=9),
            'gt_labels_3d': np.array([0], dtype=np.int64),
            'gt_names': ['car'],
            'attr_labels': np.zeros((1, 20), dtype=np.float32) 
        }

def run_aligned_inference():
    sys.path.append('.')
    import projects.mmdet3d_plugin
    
    cfg = Config.fromfile('projects/configs/VAD/VAD_tiny_e2e.py')
    test_pipeline = [s for s in cfg.data.test.pipeline if s['type'] not in ['LoadPointsFromFile', 'LoadAnnotations3D']]

    # 这里的步骤要对齐
    dataset = VADCARLADataset('carla_bridge/step_debug_info/step_0510.pkl', test_pipeline)
    data = dataset.prepare_test_data(0)

    # scatter 参数必须是整数列表 [0]
    data = collate([data], samples_per_gpu=1)
    data = scatter(data, [0])[0] 

    model = init_model('projects/configs/VAD/VAD_tiny_e2e.py', 'ckpts/VAD_tiny.pth', device='cuda:0')
    model.eval()
    
    print("🚀 接口对齐成功，开始点火推理...")
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    print("\n" + "★"*30)
    # 规划轨迹通常在 ego_fut 中
    traj = result[0]['pts_bbox']['ego_fut_preds'][0].cpu().numpy()
    print(f"🔥 奶奶！大功告成，规划轨迹出来了：\n{traj}")
    print("★"*30)

if __name__ == '__main__':
    run_aligned_inference()