#!/usr/bin/env bash
# Linux / WSL + NVIDIA GPU. Usage: bash scripts/setup_env.sh [environment-name]
set -euo pipefail
ENV_NAME="${1:-vad_carla}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v conda >/dev/null || { echo '请先安装 Miniconda，并确保 conda 在 PATH 中。'; exit 1; }
command -v nvcc >/dev/null || { echo '编译 mmdet3d 需要 CUDA Toolkit 11.7（nvcc），请先安装并配置 CUDA_HOME/PATH。'; exit 1; }
command -v gcc >/dev/null || { echo '请先安装 build-essential。'; exit 1; }
command -v git >/dev/null || { echo '请先安装 git。'; exit 1; }
if ! nvcc --version | grep -q 'release 11.7'; then
    echo '本脚本使用 PyTorch cu117，请使用 CUDA Toolkit 11.7。'; exit 1
fi
# 创建新环境，避免覆盖已有的 vad 环境。
conda create -y -n "$ENV_NAME" python=3.8 pip
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m pip install 'pip<25' 'setuptools==59.5.0' 'wheel<0.43'
python -m pip install 'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install 'numpy==1.23.5'
python -m pip install 'mmcv-full==1.7.1' -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
python -m pip install -r "$PROJECT_ROOT/requirements-carla.txt"
# mmdet3d 包含 CUDA 扩展，需要本地编译；源码保留在项目外。
DEPS_ROOT="${VAD_DEPS_ROOT:-$HOME/.local/src/vad-carla}"
mkdir -p "$DEPS_ROOT"
if [[ -e "$DEPS_ROOT/mmdetection3d" ]]; then
    echo "依赖源码目录已存在：$DEPS_ROOT/mmdetection3d；请设置 VAD_DEPS_ROOT 为新的目录后重试。"; exit 1
fi
git clone --depth 1 --branch v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git "$DEPS_ROOT/mmdetection3d"
python -m pip install --no-build-isolation "$DEPS_ROOT/mmdetection3d"
python -m pip check
cd "$PROJECT_ROOT"
python -c 'import torch, carla, pygame, mmcv, mmdet3d; import projects.mmdet3d_plugin; print("依赖导入成功；CUDA 可用:", torch.cuda.is_available())'
echo "安装完成。运行前执行：conda activate $ENV_NAME"
