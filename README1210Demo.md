# 📘 VAD-Tiny 基准算法运行操作手册

适用环境： WSL2 (Ubuntu) + NVIDIA RTX 4060Ti

数据版本： nuScenes Mini (v1.0-mini)

模型版本： VAD Tiny (E2E)

------

### 一、 准备工作 (每次开机必做)

1. **打开终端：** 打开 VS Code，点击上方菜单“终端” -> “新建终端”。

2. **进入环境：** 确保命令行前面有 `(vad)` 字样。如果没有，输入：

   Bash

   ```
   conda activate vad
   ```

3. **回到代码主目录：**

   Bash

   ```
   cd ~/VAD
   ```

------

### 二、 核心输入与输出 (Input & Output)

在您运行代码前，心里要有这张图：

- **输入 (原材料):**
  - **地图/照片数据:** 存放在 `data/nuscenes/` (包含 maps, samples, sweeps, pkl索引)。
  - **模型大脑 (权重):** `ckpts/VAD_tiny.pth`。
  - **控制参数 (配置):** `projects/configs/VAD/VAD_tiny_e2e.py`。
- **输出 (成品):**
  - **中间结果 (.pkl):** 包含所有检测框、轨迹预测的数字数据。
  - **最终结果 (Video):** 可视化的视频或图片文件夹。

------

### 三、 运行步骤 (Step-by-Step)

整个过程分为**两步走**：先推理算数，再画图看片。

#### 第一步：模型推理 (Inference)

**目的：** 让模型看一遍数据，生成推理结果文件。

在终端输入以下命令（直接复制）：

Bash

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/VAD/VAD_tiny_e2e.py ckpts/VAD_tiny.pth --eval bbox --tmpdir tmp_run
```

- **运行现象：** 出现进度条 `[>>>>>>>>] 81/81`，最后显示 `mAP` 等分数表格。

- 关键结果： 终端最后几行会告诉您结果保存的位置，通常长这样：

  test/VAD_tiny_e2e/Wed_Dec_10_.../pts_bbox/results_nusc.pkl

#### 第二步：结果提取 (Copy)

**目的：** 因为上面生成的路径带时间戳太长了，我们把它复制出来改个短名字，方便下一步操作。

在终端输入（注意中间的星号 `*` 会自动匹配日期文件夹）：

Bash

```
cp test/VAD_tiny_e2e/*/pts_bbox/results_nusc.pkl my_result.pkl
```

*现在，VAD根目录下就有了一个叫 `my_result.pkl` 的文件。*

#### 第三步：可视化生成 (Visualization)

**目的：** 把枯燥的数据变成视频。

在终端输入：

Bash

```
python tools/analysis_tools/visualization.py --result-path my_result.pkl --save-path my_video_output
```

- **运行现象：** 进度条走完 `100%`。

------

### 四、 查看结果

**目的：** 在 Windows 窗口里欣赏成果。

在终端输入：

Bash

```
explorer.exe my_video_output
```

- 弹出的文件夹里就是生成的视频和图片。

------

### 💡 奶奶的备忘录 (Tips)

1. **如果想改策略：** 您以后做“主动破局”的研究，主要就是去修改代码里的逻辑，改完后，**重复运行上面的“第一步”到“第三步”**，就能看到新策略的效果了。
2. **显存不够：** 如果以后突然报错 `Out of memory`，请去配置文件 `projects/configs/VAD/VAD_tiny_e2e.py` 里，把 `samples_per_gpu=1` 确认一下，或者把图片尺寸改小。

Next Step:

至此，复现 (Reproduction) 阶段圆满结束！

接下来我们要进入 创新 (Innovation) 阶段。您可以休息两天，然后我们可以开始讨论：如何在现有的这个代码里，植入您的“主动破局”逻辑？ (比如在哪里加几行代码，能让车在拥堵时变得更激进一点)。