#!/usr/bin/env bash
# Activate the target conda environment first.
# Usage: bash scripts/install_carla_api.sh /path/to/carla-*.egg
set -euo pipefail
if [[ $# -ne 1 || ! -f "$1" ]]; then
    echo '用法：bash scripts/install_carla_api.sh /实际路径/carla-版本-python版本-linux-x86_64.egg'
    exit 1
fi
python - "$1" <<'PY'
from pathlib import Path
import shutil
import sys
import sysconfig
import zipfile

source = Path(sys.argv[1]).resolve()
if not source.name.endswith('.egg') or not zipfile.is_zipfile(source):
    raise SystemExit('需要 CARLA 发布包中的 Python API .egg 文件。')
with zipfile.ZipFile(source) as archive:
    if 'carla/__init__.py' not in archive.namelist():
        raise SystemExit('该 egg 不包含 carla Python 模块。')
site = Path(sysconfig.get_paths()['purelib'])
target = site / 'carla_client_eggs' / source.name
target.parent.mkdir(parents=True, exist_ok=True)
if source != target:
    shutil.copy2(source, target)
(site / 'carla.pth').write_text(str(target) + '\n')
print('客户端已复制到当前 Python 环境：', target)
PY
python -c 'import carla; print("CARLA 导入成功：", carla.__file__)'
