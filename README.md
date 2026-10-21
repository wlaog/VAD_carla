# VAD × CARLA

将开源 VAD Tiny 模型接入 CARLA：采集六路相机图像与自车状态，完成感知、预测与轨迹规划，再通过 PID 将轨迹转换为车辆控制。支持手动驾驶、CARLA 自动驾驶、VAD 控车及输入录制、离线 BEV 可视化。

当前 CARLA 主入口使用 **VAD Tiny**；`VADv2/` 为保留的上游代码，未接入此运行链路。本项目已在 Windows CARLA 服务端 + WSL Python 客户端上跑通。

## Demo 展示

**[打开场景与输出展示页 → demo.html](demo.html)**

包含 `DemoMain.png` 运行截图、CARLA 与 nuScenes 的 GIF 短片、CARLA 六相机 + BEV 场景、nuScenes 六视角轨迹投影及输入检查图。支持切换场景、相机视角、暂停动画和点击图片放大。

图片与 GIF 均内嵌在这一个 HTML 文件中，删除原始输出目录后仍能离线查看；无需安装环境或运行模型。GitHub 仓库中的 HTML 默认显示源码，下载 `demo.html` 后用浏览器打开即可，也可通过 GitHub Pages 托管。

## Quick Start

以下命令均在仓库根目录执行。

### 1. Python 环境

已有可运行的环境时直接执行：

```bash
conda activate vad
```

新机器可使用安装脚本（Linux / WSL、Miniconda、NVIDIA GPU）：

```bash
# 预先安装 git、build-essential、CUDA Toolkit 11.7，并配置 nvcc / CUDA_HOME。
bash scripts/setup_env.sh vad_carla
conda activate vad_carla
```

脚本创建独立环境，安装 PyTorch、OpenMMLab 栈、CARLA Python API 和可视化依赖，并编译 MMDetection3D。核心版本来自当前运行环境：Python 3.8、PyTorch 1.13.1+cu117、MMCV 1.7.1、MMDetection 2.28.2、MMDetection3D 1.0.0rc6。其他依赖见 [requirements-carla.txt](requirements-carla.txt)。

**安装脚本尚未在全新环境验证**；旧 CUDA 扩展可能需要适配本机编译器。原 `requirements.txt` 是历史环境清单，不是本项目的一键 pip 安装入口。

### 2. 准备模型权重

从 [VAD Tiny 官方权重链接](https://drive.google.com/file/d/1KgCC_wFqPH0CQqdr6Pp2smBX5ARPaqne/view?usp=sharing)下载并命名为 `VAD_tiny.pth`，放到：

```text
ckpts/VAD_tiny.pth
```

权重约 463 MB，不随 Git 提交。更多可重新下载的资源如下，权重链接来自 [上游模型列表](https://github.com/hustvl/VAD#models)：

| 资源 | 下载入口 | 用途 / 放置位置 |
| --- | --- | --- |
| VAD Tiny | [Google Drive](https://drive.google.com/file/d/1KgCC_wFqPH0CQqdr6Pp2smBX5ARPaqne/view?usp=sharing) | 在线主入口必需，`ckpts/VAD_tiny.pth` |
| VAD Base | [Google Drive](https://drive.google.com/file/d/1FLX-4LVm4z-RskghFbxGuYlcYOQmV5bS/view?usp=sharing) | 部分历史可视化脚本使用，`ckpts/VAD_base.pth` |
| ResNet50 预训练权重 | [PyTorch 下载](https://download.pytorch.org/models/resnet50-19c8e357.pth) | 上游安装文档使用的骨干权重，可选；单独放在 `ckpts/` 不等于已配置 Torch 缓存 |
| nuScenes 数据集 | [官网下载页](https://www.nuscenes.org/nuscenes) / [上游数据准备说明](https://github.com/hustvl/VAD/blob/main/docs/prepare_dataset.md) | 仅离线流程需要，按说明准备到 `data/nuscenes/`；下载需登录 |
| CARLA 0.9.10.1 | [官方版本下载页（Windows / Linux）](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1) | 当前演示所用服务端版本，在仓库外安装 |

CARLA 录制输入与中间结果是本地实验产物，无法通过下载权重恢复；本仓库仅在 `demo.html` 中保留精选展示。在线推理入口会禁用骨干预训练初始化，完整 VAD 权重准备好后即可加载。

### 3. 启动 CARLA 服务端

先启动 CARLA 并等待地图加载，再修改 `carla_bridge/run.py` 顶部的连接配置：

```python
HOST = '127.0.0.1'; PORT = 2000; TM_PORT = 8000
```

当前 Windows + WSL 配置已验证可通过 `127.0.0.1:2000` 连接。其他部署方式填写从 Python 客户端可达的服务端地址，并放通 RPC、传感器流等所需端口；Traffic Manager 使用 `8000`。客户端 API 应与服务端版本匹配。当前服务端报告 `0.9.10.1`，Python 包为 `carla==0.9.10`，仍会出现版本警告，后续建议统一版本。

可先在运行 VAD 的终端检查 TCP 连接：

```bash
python -c "import socket; s=socket.create_connection(('127.0.0.1',2000),timeout=5); print('连接成功'); s.close()"
```

### 4. 运行与操作

```bash
python carla_bridge/run.py
```

等待权重加载完成，点击 Pygame 窗口获得键盘焦点。初始模式为暂停。

| 按键 | 功能 |
| --- | --- |
| A / Enter | CARLA 自动驾驶 |
| V | VAD 轨迹规划 + PID 控车 |
| M | 手动模式：方向键驾驶，空格手刹 |
| R | 重新生成交通车辆 |
| Q | 退出 |

默认创建 30 辆交通车、10 个行人，仿真步长 0.1 秒。部分行人因出生点碰撞生成失败时可继续运行。模型跨数据域应用于 CARLA，当前演示不代表已通过驾驶能力评测；导航指令目前在适配器中固定设置。

## 目录与运行依赖

```text
carla_bridge/
├── run.py                   # 在线仿真入口、传感器与模式控制
├── vad_carla_bridge.py      # 输入适配、VAD 推理、PID 控车
├── run_play.py              # 录制输入离线推理及 BEV 输出（Tiny）
├── visualization.py         # 另一套 BEV 可视化（当前选择 Base 权重）
├── input_compare.py         # CARLA / nuScenes 输入对比
└── InputDict_check/         # 调试工具与本地录制数据
projects/
├── configs/                 # 模型配置及继承配置
└── mmdet3d_plugin/           # VAD 模型及自定义数据处理
ckpts/                       # 本地权重，不提交
demo.html                    # 精选原图与轻量 GIF，单文件离线展示，随 Git 提交
scripts/setup_env.sh         # 环境安装脚本
requirements-carla.txt       # CARLA 演示依赖
VADv2/                      # 保留的上游 VADv2 代码
```

在线运行不直接使用 nuScenes 数据集，也不依赖同级 `~/carla/` 或 `~/carla_source/` 目录。Python 环境与 CARLA 服务端需要单独准备；安装脚本将 MMDetection3D 源码放在 `~/.local/src/vad-carla/`，可通过 `VAD_DEPS_ROOT` 修改。

运行时图片与每步输入保存到 `carla_bridge/InputDict_check/input_dictC/`，会持续占用磁盘。离线可视化：

```bash
python carla_bridge/run_play.py
```

该脚本默认从第 9 步读取录制数据，图片路径需仍有效；输出到录制目录下的 `bev_vis/`。其他可视化脚本可能使用不同权重，请先检查脚本末尾配置。

## Git 提交说明

源码、配置、安装脚本及 `demo.html` 纳入版本控制；权重、数据集、CARLA 原始录制、评测结果、批量可视化输出和本机 IDE 配置已加入 `.gitignore`。展示文件内嵌精选原图和 GIF，不依赖被忽略的目录。历史实验说明见 [README1210Demo.md](README1210Demo.md)，其中为 nuScenes 离线流程。

## 参考

- [上游 VAD 仓库及原始 README](https://github.com/hustvl/VAD)：模型、权重下载与论文资料。
- [CARLA](https://github.com/carla-simulator/carla)

本项目基于上游 VAD 修改，保留原有 [LICENSE](LICENSE) 及源码版权声明。
