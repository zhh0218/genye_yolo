# S8 RGB-D YOLO11-seg 网络模型结构报告

本文用于把当前项目中的 `coord_attv2-s8` 网络结构完整交给 ChatGPT 或其他模型阅读、分析和继续优化。报告基于本仓库当前源码与训练产物整理，重点说明模型结构、前向传播路径、关键模块、数据输入方式和源码位置。

## 1. 一句话概括

S8 是一个基于 Ultralytics YOLO11-seg 改造的 RGB-D 双分支实例分割模型。它保留 YOLO11-seg 的 RGB 主干和分割检测头，额外增加一个与 RGB backbone 同构的 depth backbone；depth 分支先经过轻量级 top-down FPN 增强多尺度上下文，然后在 P3、P4、P5 三个尺度与 RGB 特征融合，融合模块使用 `CoordAttV2`，其中 P3 额外启用 Haar wavelet 引导，最后使用 YOLO11-seg 的 Segment head 输出 bbox 和 mask。

核心结构可以简写为：

```text
RGB image   -> RGB backbone   -> RGB P3/P4/P5 ----┐
Depth image -> Depth backbone -> Depth P3/P4/P5 -> DepthLightFPN
                                                    │
P3: RGB P3 + Depth P3 -> WaveletGuidedCoordAttV2 ---┤
P4: RGB P4 + Depth P4 -> CoordAttV2 ----------------┤
P5: RGB P5 + Depth P5 -> CoordAttV2 ----------------┘
              -> YOLO11 neck/head -> Segment(P3, P4, P5) -> boxes + masks
```

## 2. 模型身份与训练产物

模型名称：`coord_attv2-s8`

推荐描述名：

```text
YOLO11-seg-RGBD-CoordAttV2-P3Wavelet-DepthFPN
```

主要配置文件：

```text
ultralytics/cfg/models/11/yolo11-seg.yaml
```

训练产物目录：

```text
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8
```

权重：

```text
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/last.pt
```

训练配置来自：

```text
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/args.yaml
```

关键训练参数：

```yaml
task: segment
model: /workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg.yaml
data: /workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml
epochs: 300
batch: 36
imgsz: 640
device: 0,1,2,3
optimizer: MuSGD
pretrained: /workspace/ultralytics-main_for_genye/yolo11l.pt
amp: true
```

从 `results.csv` 最后一轮看，epoch 300 的主要指标为：

```text
Box mAP50      = 0.91062
Box mAP50-95   = 0.81455
Mask mAP50     = 0.91523
Mask mAP50-95  = 0.80208
```

历史 README 中还记录过 `best Mask mAP50-95 = 0.80303`，以及重新验证 `best.pt` 约为 `Mask mAP50-95 = 0.79863`。

## 3. YAML 顶层开关

S8 的核心开关位于 `ultralytics/cfg/models/11/yolo11-seg.yaml`：

```yaml
scale: "s"
end2end: False
reg_max: 1
rgbd_fusion: "coord_att_v2"
p3_wavelet_guided: True
p3_wavelet_HF: False
p3_wavelet_adaptive: False
p4_wavelet_guided: False
depth_fpn: True
modality_adaptive_gate: False
rgbd_reliability_gate: False
cls01_weight: 1.0
ol_iou: False
prog_loss: False
stal: False
```

含义：

- `scale: 's'`：使用 YOLO11 small 规模，宽度系数 0.50，深度系数 0.50。
- `reg_max: 1`：关闭 DFL，bbox 使用直接回归，因此训练日志里的 `dfl_loss` 为 0 是正常现象。
- `rgbd_fusion: 'coord_att_v2'`：P3/P4/P5 的默认 RGB-D 融合模块使用 `RGBDCoordAttV2`。
- `p3_wavelet_guided: True`：P3 小目标尺度使用 `RGBDWaveletGuidedCoordAttV2`，用 depth 的 Haar 高频边缘门控引导融合。
- `p4_wavelet_guided: False`：P4 不使用 wavelet 引导。
- `depth_fpn: True`：depth P3/P4/P5 在融合前先进入 `DepthLightFPN`。
- `modality_adaptive_gate: False`：原始 S8 不启用后续 spatial illumination gate。
- `rgbd_reliability_gate: False`：原始 S8 不启用额外 RGB-D reliability gate。

注意：`configs/yolo11-seg-spatialillum-learnblend-auxoff-xinjiang1500.yaml` 是 S8 后续增强配置，启用了 `modality_adaptive_gate: True`、`gate_mode: spatial_illum` 和 `fusion_learnable_blend: True`。如果只分析原始 S8，应以 `ultralytics/cfg/models/11/yolo11-seg.yaml` 为准。

## 4. 输入数据与 Dataset

模型训练和验证使用 RGB-D 成对输入：

```text
batch["img"]       -> RGB 图像张量
batch["depth_img"] -> depth 图像张量
```

数据配置文件：

```text
data-ubuntu-genye.yaml
```

内容为：

```yaml
train: /workspace/Datasets/final/images/train
val: /workspace/Datasets/final/images/val
test: /workspace/Datasets/final/images/test

nc: 2
names:
  0: 00
  1: 01
```

depth 路径可以显式写成 `depth_train`、`depth_val`、`depth_test`，也可以由代码自动从 RGB 路径推断：把路径中的 `/images/` 替换成 `/depth/`。因此推荐数据目录为：

```text
Datasets/final/
  images/train/xxx.jpg
  images/val/xxx.jpg
  depth/train/xxx.png
  depth/val/xxx.png
  labels/train/xxx.txt
  labels/val/xxx.txt
```

RGB 和 depth 通过文件 stem 配对，例如：

```text
images/train/box_0001.jpg
depth/train/box_0001.png
labels/train/box_0001.txt
```

相关源码：

- `ultralytics/data/build.py`：`build_yolo_dataset()` 根据 `depth_path` 决定是否使用 `YOLORGBDDataset`。
- `ultralytics/data/dataset.py`：`YOLORGBDDataset` 负责 depth 文件配对、读取、尺寸对齐，并把 `depth_img` 放入 label dict。
- `ultralytics/data/augment.py`：所有涉及 mosaic、mixup、copy-paste、random perspective、flip、letterbox、format 的变换都同步处理 `depth_img`，避免 RGB 与 depth 错位。

## 5. 主体网络结构

### 5.1 RGB backbone

RGB backbone 是 YOLO11-seg 的主干，层号 0 到 10：

```text
0  Conv      -> P1/2
1  Conv      -> P2/4
2  C3k2
3  Conv      -> P3/8
4  C3k2      -> RGB P3 feature
5  Conv      -> P4/16
6  C3k2      -> RGB P4 feature
7  Conv      -> P5/32
8  C3k2
9  SPPF
10 C2PSA     -> RGB P5 feature
```

实际用于融合的 RGB 特征层：

```text
P3: y[4]
P4: y[6]
P5: y[10]
```

在 `scale='s'` 下，通道经过 width multiplier 0.50 缩放，典型融合通道约为：

```text
P3: 128 channels
P4: 256 channels
P5: 512 channels
```

### 5.2 YOLO11 segmentation head

RGB 分支的 head 层号 11 到 23：

```text
11 Upsample
12 Concat with backbone P4 y[6]
13 C3k2

14 Upsample
15 Concat with backbone P3 y[4]
16 C3k2 -> head P3/8

17 Conv downsample
18 Concat with head P4 y[13]
19 C3k2 -> head P4/16

20 Conv downsample
21 Concat with backbone P5 y[10]
22 C3k2 -> head P5/32

23 Segment([16, 19, 22], nc, nm=32, npr=256)
```

由于 S8 会在进入 head 前替换 `y[4]`、`y[6]`、`y[10]` 为融合后的 RGB-D 特征，所以后续 YOLO11 segmentation head 不需要知道 depth 分支存在。

### 5.3 Depth backbone

YAML 中额外定义了 `backboneD`，结构与 RGB backbone 基本同构：

```text
0  Conv
1  Conv
2  C3k2
3  Conv
4  C3k2      -> Depth P3 feature
5  Conv
6  C3k2      -> Depth P4 feature
7  Conv
8  C3k2
9  SPPF
10 C2PSA     -> Depth P5 feature
```

在 PyTorch `self.model` 中，depth backbone 不从 0 开始，而是追加到 RGB backbone + head 后面。因为 YAML 中 RGB backbone 有 11 层、head 有 13 层，所以：

```text
RGB backbone: 0..10
RGB head:     11..23
Depth start:  24
Depth end:    34
```

源码中通过以下方式自动计算：

```python
_n_backbone = len(self.yaml.get("backbone", []))
_n_head = len(self.yaml.get("head", []))
_depth_offset = _n_backbone + _n_head
_n_depth = len(self.yaml.get("backboneD", []))
self._depth_range = (_depth_offset, _depth_offset + _n_depth)
```

Depth 分支对应关系：

```text
Depth local 4  -> absolute layer 28 -> Depth P3
Depth local 6  -> absolute layer 30 -> Depth P4
Depth local 10 -> absolute layer 34 -> Depth P5
```

## 6. 额外模块插入位置

`parse_model()` 先解析：

```text
backbone + head + backboneD
```

然后再根据 YAML 开关追加 RGB-D 模块：

```text
如果 depth_fpn=True:
  append DepthLightFPN

append P3 fusion module
append P4 fusion module
append P5 fusion module

如果 rgbd_reliability_gate=True:
  append 3 个 RGBDRelativeReliabilityGate

如果 rgbd_depth_aux_loss=True:
  append RGBDDepthAuxHead
```

原始 S8 中：

```text
depth_fpn=True
rgbd_reliability_gate=False
rgbd_aux_loss=False
```

所以追加顺序是：

```text
35 DepthLightFPN
36 P3 RGBDWaveletGuidedCoordAttV2
37 P4 RGBDCoordAttV2
38 P5 RGBDCoordAttV2
```

注意：层号 35-38 是根据当前 YAML 层数推导出来的，用于理解当前 S8；如果 YAML 层数变化，应以源码自动计算结果为准。

## 7. 前向传播流程

核心前向逻辑在：

```text
ultralytics/nn/tasks.py
BaseModel._predict_once(self, x, xd, ...)
```

其中：

```text
x  = RGB tensor
xd = depth tensor
```

训练时，`BaseModel.loss()` 会取：

```python
depth = batch.get("depth_img")
preds = self.forward(batch["img"], depth)
```

验证时，`validator.py` 会调用：

```python
preds = model(batch["img"], batch.get("depth_img"), augment=augment)
```

S8 前向分为三阶段：

### 阶段 1：单独跑 depth backbone

```text
xd -> depth backbone absolute layers 24..34
保存 depth local 4/6/10，也就是 Depth P3/P4/P5
```

得到：

```text
_depth_save_map[4]  = depth P3
_depth_save_map[6]  = depth P4
_depth_save_map[10] = depth P5
```

### 阶段 2：跑 RGB backbone

```text
x -> RGB backbone layers 0..10
保存 y[4], y[6], y[10]
```

得到：

```text
y[4]  = RGB P3
y[6]  = RGB P4
y[10] = RGB P5
```

### 阶段 3：Depth FPN + RGB-D 融合

如果 `depth_fpn=True`：

```python
p3d, p4d, p5d = DepthLightFPN(depth_P3, depth_P4, depth_P5)
```

然后三个尺度分别融合：

```text
y[4]  = P3_fusion(y[4],  depth_P3)
y[6]  = P4_fusion(y[6],  depth_P4)
y[10] = P5_fusion(y[10], depth_P5)
```

原始 S8 的三个 fusion module：

```text
P3: RGBDWaveletGuidedCoordAttV2
P4: RGBDCoordAttV2
P5: RGBDCoordAttV2
```

最后从融合后的 `y[10]` 继续跑 YOLO head：

```text
x = y[10]
run layers 11..23
return Segment output
```

## 8. DepthLightFPN 结构

源码：

```text
ultralytics/nn/tasks.py
class DepthLightFPN
```

结构：

```python
p4 = reduce5to4(cat(upsample(p5), p4))
p3 = reduce4to3(cat(upsample(p4), p3))
return p3, p4, p5
```

作用：

- 把 depth P5 的语义信息上采样并注入 P4。
- 再把增强后的 P4 上采样并注入 P3。
- P5 保持原值。
- 这样在 RGB-D 融合前，depth P3/P4 会带有更强的多尺度上下文。

结构图：

```text
Depth P5 --upsample--┐
                     concat -> 1x1 Conv -> new Depth P4 --upsample--┐
Depth P4 ------------┘                                               concat -> 1x1 Conv -> new Depth P3
Depth P3 ------------------------------------------------------------┘
```

## 9. CoordAttV2 融合模块

源码：

```text
ultralytics/nn/tasks.py
class CoordAttV2
class RGBDCoordAttV2
```

`RGBDCoordAttV2` 只是一个适配器，真正做融合的是 `CoordAttV2`：

```python
x = torch.cat((rgb, depth), dim=1)
```

如果输入单尺度特征通道为 `C`，则 concat 后通道为 `2C`。

CoordAttV2 的核心计算：

```text
1. concat RGB 和 depth: x = [rgb, depth]
2. 分别沿 H 和 W 做 coordinate pooling
3. 拼接 H/W pooled features
4. 经过 1x1 Conv + BN + h_swish
5. 分成 height attention 和 width attention
6. 得到 a_h 和 a_w
7. 对 concat 特征做坐标注意力: x * a_h * a_w
8. 通过 proj 投影回 C 通道
9. 残差输出: rgb + out
```

原始 S8 没有开启 `adaptive_gate` 和 `learnable_blend`，所以输出为：

```python
return rgb + out
```

这意味着：

- RGB 是主路径。
- depth 通过 concat attention 产生补充特征 `out`。
- 模块初衷是保留 RGB 表达，同时用 depth 引导增强或校正。

## 10. P3 Wavelet-Guided CoordAttV2

源码：

```text
ultralytics/nn/tasks.py
class RGBDWaveletGuidedCoordAttV2
```

只有 P3 开启：

```yaml
p3_wavelet_guided: True
p4_wavelet_guided: False
```

P3 是小目标和边界最敏感的尺度，所以 S8 在 P3 上加入 Haar wavelet 引导。

模块流程：

```text
depth P3 -> Haar split -> LL, LH, HL, HH
LL -> low_proj -> resize to RGB P3 size -> depth_low
LH/HL/HH -> concat -> hf_reduce -> hf_refine -> edge_gate
base = CoordAttV2(rgb P3, depth_low)
output = base + beta * base * sigmoid(edge_gate)
```

其中：

```python
self.beta = nn.Parameter(torch.ones(1) * 0.1)
```

理解：

- `LL` 表示低频/主体轮廓信息。
- `LH/HL/HH` 表示高频边缘、纹理和变化信息。
- `depth_low` 作为 CoordAttV2 的 depth 输入。
- 高频分量生成 `edge_gate`，对融合后的 base 特征做边缘增强。
- `beta` 是可学习强度，初始为 0.1，避免一开始过度扰动主干特征。

## 11. Segment head 输出

最终输出层：

```yaml
[[16, 19, 22], 1, Segment, [nc, 32, 256]]
```

含义：

- 输入三个尺度：P3/8、P4/16、P5/32。
- `nc`：类别数，训练数据里覆盖为 2。
- `nm=32`：mask coefficients 数量。
- `npr=256`：prototype mask 通道/原型维度。
- `reg_max=1`：关闭 DFL。

输出包括：

```text
box predictions
class predictions
mask coefficients
mask prototypes
```

训练和验证指标里：

```text
metrics/mAP50(B), metrics/mAP50-95(B) -> box 指标
metrics/mAP50(M), metrics/mAP50-95(M) -> mask 指标
```

该项目以实例分割为主，因此优先关注 `metrics/mAP50-95(M)`。

## 12. 关键源码索引

建议让 ChatGPT 重点看这些位置：

```text
ultralytics/cfg/models/11/yolo11-seg.yaml
  - 顶层开关
  - RGB backbone
  - YOLO11 segmentation head
  - depth backboneD

ultralytics/nn/tasks.py
  - BaseModel._predict_once
  - DetectionModel.__init__
  - CoordAttV2
  - RGBDCoordAttV2
  - RGBDWaveletGuidedCoordAttV2
  - DepthLightFPN
  - parse_model

ultralytics/data/build.py
  - build_yolo_dataset

ultralytics/data/dataset.py
  - YOLORGBDDataset
  - get_depth_files
  - load_depth_image
  - get_image_and_label
  - collate_fn

ultralytics/data/augment.py
  - 所有 depth_img 同步增强逻辑

ultralytics/models/yolo/detect/train.py
  - preprocess_batch

ultralytics/models/yolo/detect/val.py
  - preprocess

ultralytics/engine/validator.py
  - model(batch["img"], batch.get("depth_img"))
```

## 13. 与普通 YOLO11-seg 的区别

普通 YOLO11-seg：

```text
RGB -> backbone -> neck/head -> Segment
```

S8：

```text
RGB   -> RGB backbone ----------------------┐
Depth -> Depth backbone -> DepthLightFPN ---┤
                                           P3/P4/P5 fusion
                                             ↓
                         YOLO11 segmentation head -> Segment
```

主要改动：

1. 输入从单 RGB 变成 RGB + depth。
2. 增加 `backboneD`，depth 分支与 RGB backbone 同构。
3. 增加 `DepthLightFPN`，在融合前增强 depth 多尺度表达。
4. P3/P4/P5 进行跨模态融合。
5. P3 使用 wavelet 高频边缘门控，强化小目标/边界。
6. 原 YOLO11 Segment head 基本保留，只接收融合后的 RGB-D 特征。

## 14. 可能的疑点与注意事项

1. `backboneD` 的第一层输入通道在 `parse_model()` 中被特殊处理为 3。depth 图片如果是单通道，会在 dataset 中复制成 3 通道。

2. `reg_max=1` 导致 DFL 关闭，日志中 `train/dfl_loss` 和 `val/dfl_loss` 为 0 是设计结果。

3. 原始 S8 没启用 `modality_adaptive_gate`，所以 `CoordAttV2` 输出是 `rgb + out`，不是空间自适应的 `alpha * rgb + (1-alpha) * depth + out`。

4. 原始 S8 没启用 `rgbd_reliability_gate`，因此没有在 fusion 前额外缩放 RGB/depth 相对可靠性。

5. 单图推理不能只走普通 `yolo predict` 的 RGB 输入路径；需要确保 depth 输入能传入 `model(rgb, depth)`。

6. 数据增强必须同步处理 RGB 和 depth。当前仓库已在 `augment.py` 中对 `depth_img` 做了同步 mosaic、flip、letterbox 等处理。

## 15. 可直接发给 ChatGPT 的提问模板

可以把下面这段和本报告一起发给 ChatGPT：

```text
你现在要分析一个基于 YOLO11-seg 的 RGB-D 实例分割模型 S8。请根据报告理解模型结构，并从以下角度给出建议：

1. 这个 S8 网络的 RGB backbone、depth backbone、DepthLightFPN、P3/P4/P5 融合和 Segment head 的信息流是否合理？
2. P3 使用 Haar wavelet 引导、P4/P5 使用普通 CoordAttV2 的设计是否符合小目标和边界增强的目标？
3. CoordAttV2 当前输出为 rgb + out，没有启用 adaptive gate。是否建议改成空间自适应 gate？可能风险是什么？
4. DepthLightFPN 只增强 P3/P4、不改变 P5 是否合理？有没有更轻量或更稳的替代方案？
5. 如果当前 Mask mAP50-95 约为 0.80，下一步更值得做结构消融、损失函数调整、数据增强调整还是推理后处理调整？
6. 请给出适合论文/报告描述的网络结构文字、模块公式和消融实验设计。
```

## 16. 面向论文/报告的简洁表述

S8 模型在 YOLO11-seg 的基础上构建 RGB-D 双分支实例分割框架。RGB 图像和深度图分别输入两个结构同构的 backbone，提取 P3、P4、P5 三个尺度的特征。为增强深度分支的多尺度语义表达，模型在融合前引入轻量级 top-down Depth FPN，将 P5 语义逐级传递至 P4 和 P3。随后，模型在三个尺度进行 RGB-D 特征融合，其中 P4 和 P5 采用 CoordAttV2 融合模块，P3 采用 Haar wavelet 引导的 CoordAttV2 模块，以利用深度图高频分量增强小目标边界和局部结构。融合后的 P3、P4、P5 特征输入 YOLO11-seg 的分割头，最终输出目标检测框、类别和实例掩码。
