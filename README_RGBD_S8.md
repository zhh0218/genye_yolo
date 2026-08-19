# YOLO11 RGB-D 包裹实例分割项目说明

本文档用于给新同学快速理解和复现当前项目。目标是让大家先跑通最终版本，再逐步阅读 RGB-D 双分支、特征融合和消融实验代码。

## 1. 项目目标

本项目基于 Ultralytics YOLO11-seg 改造，用于 RGB-D 包裹实例分割和除双相关任务。

当前最终版本为：

```text
YOLO11-seg + RGB-D 双分支 + CoordAttV2 融合 + P3 wavelet 引导 + Depth FPN
```

最终权重：

```text
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
```

推荐版本名称：

```text
YOLO11-seg-RGBD-CoordAttV2-P3Wavelet-DepthFPN
```

## 2. 最终版本指标

最终版本对应运行目录：

```text
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8
```

核心结果：

| 指标 | 数值 |
|---|---:|
| best Mask mAP50-95 | 0.80303 |
| final best.pt 验证 Mask mAP50-95 | 0.79863 |
| final best.pt 验证 Mask mAP50 | 0.91975 |
| final best.pt 验证 Box mAP50-95 | 0.82339 |
| 类别 1 Mask mAP50-95 | 0.66651 |
| 参数量 | 约 17.29M |
| 计算量 | 约 54.3 GFLOPs |

重要文件：

```text
/workspace/ultralytics-main_for_genye/LOG/genye11s_just_coord_att_v2+p3+dfpn.log
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/args.yaml
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/results.csv
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/results.png
```

## 3. 数据集结构

数据配置文件：

```text
/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml
```

当前数据路径：

```yaml
train: /workspace/Datasets/final/images/train
val: /workspace/Datasets/final/images/val
test: /workspace/Datasets/final/images/test

nc: 2
names:
  0: 00
  1: 01
```

RGB、depth、label 需要按下面结构存放：

```text
/workspace/Datasets/final/
├── images/
│   ├── train/
│   │   ├── xxx.jpg
│   │   └── ...
│   └── val/
│       ├── yyy.jpg
│       └── ...
├── depth/
│   ├── train/
│   │   ├── xxx.png
│   │   └── ...
│   └── val/
│       ├── yyy.png
│       └── ...
└── labels/
    ├── train/
    │   ├── xxx.txt
    │   └── ...
    └── val/
        ├── yyy.txt
        └── ...
```

注意：depth 图会自动通过 RGB 图路径匹配。

代码会把 RGB 路径中的：

```text
/images/
```

替换为：

```text
/depth/
```

因此 RGB 和 depth 的文件名主干必须一致，例如：

```text
images/train/box_0001.jpg
depth/train/box_0001.png
labels/train/box_0001.txt
```

如果 depth 文件名不一致，训练时会跳过样本或报错。

## 4. 环境说明

服务器上当前实验环境：

```text
Python 3.10.20
PyTorch 2.5.1 + CUDA 12.1
Ultralytics 8.3.195
GPU: RTX 4090
```

推荐先进入已有环境：

```bash
conda activate yolo26
cd /workspace/ultralytics-main_for_genye
```

如果新建环境，至少需要安装：

```bash
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy matplotlib pandas pillow tqdm pyyaml
```

如果使用项目内已修改的 Ultralytics 代码，建议在项目根目录运行，优先使用本地 `ultralytics/` 包，不要随意升级官方 ultralytics，否则可能覆盖 RGB-D 修改逻辑。

## 5. 训练

训练入口：

```text
/workspace/ultralytics-main_for_genye/train_genye.py
```

当前训练配置：

```python
model_yaml_path = "/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg.yaml"
data_yaml_path = "/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml"
pre_model_name = "/workspace/ultralytics-main_for_genye/yolo11l.pt"
```

主要超参数：

```text
imgsz=640
epochs=300
batch=36
workers=2
device=0,1,2,3
optimizer=MuSGD
amp=True
```

训练命令：

```bash
cd /workspace/ultralytics-main_for_genye
conda activate yolo26
python train_genye.py
```

后台训练并保存日志：

```bash
cd /workspace/ultralytics-main_for_genye
conda activate yolo26
nohup python train_genye.py > LOG/new_train.log 2>&1 &
tail -f LOG/new_train.log
```

说明：如果 `project/name` 已存在，Ultralytics 会自动递增运行目录，例如从 `coord_attv2-s8` 变成 `coord_attv2-s9`。

## 6. 验证

推荐验证最终权重：

```bash
cd /workspace/ultralytics-main_for_genye
conda activate yolo26
python - <<'PY'
from ultralytics import YOLO

model = YOLO("/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt")
model.val(
    data="/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml",
    imgsz=640,
    batch=36,
    device="0,1,2,3",
    split="val",
)
PY
```

验证结果重点看：

```text
metrics/mAP50(M)
metrics/mAP50-95(M)
metrics/mAP50(B)
metrics/mAP50-95(B)
```

其中 `(M)` 是 mask 分割指标，`(B)` 是 box 检测指标。当前任务以实例分割为主，优先看 `Mask mAP50-95`。

## 7. 推理

本项目是 RGB-D 双输入模型，不能简单等同于普通 RGB YOLO 推理。

推理时需要同时准备：

```text
RGB 图像
Depth 图像
```

并保证两者文件名主干一致。推荐先使用验证流程确认模型和数据加载正常，再编写或使用专用 RGB-D 推理脚本。

如果要做单图推理，建议逻辑如下：

1. 读取 RGB 图。
2. 根据 RGB 路径把 `images` 替换为 `depth`，找到 depth 图。
3. 对 RGB 和 depth 做相同 resize/letterbox。
4. RGB 输入 `batch["img"]`，depth 输入 `batch["depth_img"]`。
5. 模型前向输出实例分割结果。

不要直接只给一张 RGB 图调用普通 `yolo predict`，否则可能缺少 depth 输入，结果不符合 RGB-D 模型设计。

## 8. 核心代码导读

建议按下面顺序阅读。

### 8.1 模型配置

```text
ultralytics/cfg/models/11/yolo11-seg.yaml
```

重点看：

```yaml
rgbd_fusion: 'coord_att_v2'
p3_wavelet_guided: True
depth_fpn: True
backbone:
head:
backboneD:
```

其中：

- `backbone` 是 RGB 分支。
- `backboneD` 是 depth 分支。
- `rgbd_fusion` 控制 RGB-D 融合模块。
- `p3_wavelet_guided` 控制 P3 小目标/边缘层的 wavelet 引导。
- `depth_fpn` 控制 depth 分支轻量 FPN。

### 8.2 模型前向和融合模块

```text
ultralytics/nn/tasks.py
```

重点看：

- `_predict_once`
- `CoordAttV2`
- `RGBDCoordAttV2`
- `RGBDWaveletGuidedCoordAttV2`
- `DepthLightFPN`
- `parse_model`

整体流程：

```text
RGB backbone 提取 P3/P4/P5
Depth backbone 提取 P3/P4/P5
Depth FPN 优化 depth 多尺度特征
P3/P4/P5 分别做 RGB-D 融合
融合后的 RGB 特征进入 YOLO11-seg head
输出 box 和 mask
```

### 8.3 数据加载

```text
ultralytics/data/build.py
ultralytics/data/dataset.py
```

重点看：

- `build_yolo_dataset`
- `YOLORGBDDataset`
- `get_depth_files`
- `load_depth_image`
- `get_image_and_label`

这里实现了 RGB 和 depth 的自动配对。

### 8.4 数据增强

```text
ultralytics/data/augment.py
```

重点看所有涉及 `depth_img` 的位置。RGB 做 mosaic、flip、resize、letterbox 等增强时，depth 也要同步处理，否则 RGB 和 depth 会错位。

### 8.5 训练和验证预处理

```text
ultralytics/models/yolo/detect/train.py
ultralytics/models/yolo/detect/val.py
ultralytics/models/yolo/segment/val.py
```

重点看：

- RGB 图归一化
- depth 图归一化
- `depth_img` 如何送入模型

## 9. 消融实验结果

当前建议给新同学重点理解三组：

| run | 结构 | 结论 |
|---|---|---|
| `coord_attv2-s3` | CoordAttV2 + P3 wavelet | 旧主版本 |
| `coord_attv2-s7` | just CoordAttV2 | 更轻，效果接近 |
| `coord_attv2-s8` | CoordAttV2 + P3 wavelet + Depth FPN | 当前最强，推荐交付 |

三组结果对比：

| run | best Mask mAP50-95 | final best.pt Mask mAP50-95 | final best.pt Box mAP50-95 |
|---|---:|---:|---:|
| `s3` | 0.79678 | 0.79603 | 0.82316 |
| `s7` | 0.79749 | 0.79475 | 0.82420 |
| `s8` | 0.80303 | 0.79863 | 0.82339 |

结论：如果优先追求分割精度，使用 `s8/best.pt`。如果部署速度压力很大，可以把 `s7/best.pt` 作为轻量备选。

## 10. 常见问题

### 10.1 找不到 depth 图

现象：

```text
Depth Image Not Found
Skipping RGB images with no depth pair
```

原因：

- depth 文件缺失。
- RGB 和 depth 文件名主干不一致。
- 数据目录没有按 `images/` 和 `depth/` 对应存放。

修复：

```text
images/train/xxx.jpg
depth/train/xxx.png
labels/train/xxx.txt
```

三者主干必须一致：`xxx`。

### 10.2 出现 duplicate depth filename stem

原因：depth 目录里有多个同名主干文件，例如：

```text
xxx.png
xxx.jpg
```

修复：同一个样本只保留一个 depth 文件。

### 10.3 dfl_loss 一直是 0

这是正常现象。当前模型配置里：

```yaml
reg_max: 1
```

表示关闭 DFL，直接做 bbox regression，因此日志中 `dfl_loss=0`。

### 10.4 CUDA 显存不足

可以降低 batch：

```python
batch=16
```

或者减少 GPU 数量：

```python
device="0"
```

### 10.5 best.pt 和 last.pt 用哪个

交付和验证优先使用：

```text
weights/best.pt
```

`last.pt` 是最后一个 epoch 的权重，不一定是验证集最优。

### 10.6 deterministic warning

训练时可能看到：

```text
adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation
```

这是 PyTorch 在 deterministic 模式下的警告。当前实验中可以先忽略，但正式复现实验时需要记录该信息。

## 11. 新同学学习路线

建议按下面顺序学习：

1. 阅读本 README，明确任务和最终版本。
2. 跑一次 `best.pt` 验证，确认环境和数据没问题。
3. 看 `data-ubuntu-genye.yaml`，理解数据路径。
4. 看 `YOLORGBDDataset`，理解 RGB-depth 配对。
5. 看 `yolo11-seg.yaml`，理解双分支结构。
6. 看 `tasks.py` 的 `_predict_once`，理解 P3/P4/P5 融合流程。
7. 跑一个小 epoch 训练，确认训练流程。
8. 改一个融合开关，做一次小消融。

## 12. 交付建议

给别人使用时，至少打包以下内容：

```text
README_RGBD_S8.md
data-ubuntu-genye.yaml
train_genye.py
ultralytics/cfg/models/11/yolo11-seg.yaml
ultralytics/nn/tasks.py
ultralytics/data/build.py
ultralytics/data/dataset.py
ultralytics/data/augment.py
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/results.csv
LOG/genye11s_just_coord_att_v2+p3+dfpn.log
```

如果只给学习，不建议一开始给太多历史实验目录，否则容易混淆。统一说明：当前最终版就是 `s8/best.pt`。
