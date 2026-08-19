# RGB-D YOLO11-seg 源码导读

本文档用于帮助新同学快速理解本项目相对原始 Ultralytics 的核心改动。建议先阅读根目录的 `README_RGBD_S8.md`，能跑通最终版本后，再按本文档阅读源码。

当前最终版本：

```text
YOLO11-seg + RGB-D 双分支 + CoordAttV2 融合 + P3 wavelet 引导 + Depth FPN
```

最终权重：

```text
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
```

## 1. 阅读顺序

不要一开始从整个 Ultralytics 项目乱翻。建议按下面顺序阅读：

1. `train_genye.py`：训练入口和超参数。
2. `data-ubuntu-genye.yaml`：数据集路径和类别定义。
3. `ultralytics/data/build.py`：如何自动选择 RGB-D 数据集。
4. `ultralytics/data/dataset.py`：RGB 和 depth 如何配对。
5. `ultralytics/data/augment.py`：增强时 RGB 和 depth 如何同步变换。
6. `ultralytics/cfg/models/11/yolo11-seg.yaml`：模型结构和融合开关。
7. `ultralytics/nn/tasks.py`：RGB-D 前向、融合模块和模型解析。
8. `ultralytics/models/yolo/detect/train.py` 和 `val.py`：训练/验证时 depth 如何归一化并送入模型。

一句话理解流程：

```text
RGB 图和 depth 图进入 YOLORGBDDataset
    -> 同步数据增强
    -> batch 中同时包含 img 和 depth_img
    -> RGB backbone 和 depth backbone 分别提特征
    -> P3/P4/P5 做 RGB-D 融合
    -> YOLO11-seg head 输出 box 和 mask
```

## 2. 训练入口

文件：

```text
/workspace/ultralytics-main_for_genye/train_genye.py
```

作用：

- 指定模型配置文件。
- 指定数据集配置文件。
- 加载预训练权重。
- 设置训练超参数。
- 启动 YOLO 训练。

关键配置：

```python
model_yaml_path = "/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg.yaml"
data_yaml_path = "/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml"
pre_model_name = "/workspace/ultralytics-main_for_genye/yolo11l.pt"
```

最终版本训练时主要参数：

```python
imgsz=640
epochs=300
batch=36
workers=2
device="0,1,2,3"
optimizer="MuSGD"
amp=True
project="YOLOv11-RGB-D-coord_attv2-genye"
name="coord_attv2-s"
```

注意：如果已有同名运行目录，Ultralytics 会自动递增，例如 `coord_attv2-s8`、`coord_attv2-s9`。

## 3. 数据配置

文件：

```text
/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml
```

作用：

- 指定 train/val/test 的 RGB 图像路径。
- 指定类别数和类别名。

当前配置：

```yaml
train: /workspace/Datasets/final/images/train
val: /workspace/Datasets/final/images/val
test: /workspace/Datasets/final/images/test

nc: 2
names:
  0: 00
  1: 01
```

特别注意：这个 YAML 只显式写了 RGB 图像路径，没有显式写 depth 路径。depth 路径是在代码中自动推断的。

自动匹配规则：

```text
/images/ -> /depth/
```

例如：

```text
RGB:   /workspace/Datasets/final/images/train/xxx.jpg
Depth: /workspace/Datasets/final/depth/train/xxx.png
Label: /workspace/Datasets/final/labels/train/xxx.txt
```

RGB、depth、label 的文件名主干必须一致。

## 4. 数据集构建

文件：

```text
/workspace/ultralytics-main_for_genye/ultralytics/data/build.py
```

重点函数：

```python
build_yolo_dataset(...)
```

核心作用：

- 根据普通 RGB 图路径推断 depth 图路径。
- 如果存在 depth 路径，就使用 `YOLORGBDDataset`。
- 如果不存在 depth 路径，就退回普通 `YOLODataset` 或多模态数据集。

核心逻辑可以理解为：

```python
depth_path = data.get(f"depth_{mode}")
if not depth_path and isinstance(img_path, str):
    candidate = img_path.replace("/images/", "/depth/")
    if Path(candidate).exists():
        depth_path = candidate

dataset = YOLORGBDDataset if depth_path else YOLODataset
```

这部分是 RGB-D 数据进入训练流程的入口。新同学如果遇到 depth 没有被读取，优先检查这里。

## 5. RGB-D 数据配对

文件：

```text
/workspace/ultralytics-main_for_genye/ultralytics/data/dataset.py
```

重点类：

```python
class YOLORGBDDataset(YOLODataset):
```

重点函数：

```python
get_depth_files(...)
load_depth_image(...)
get_image_and_label(...)
```

### 5.1 `get_depth_files`

作用：

- 扫描 depth 目录。
- 按文件名主干和 RGB 图像配对。
- 如果某张 RGB 图没有对应 depth，会跳过并给出 warning。
- 如果 depth 目录里出现重复主干文件，会报错。

容易出错的情况：

```text
images/train/a001.jpg
depth/train/a001.png      正确

images/train/a001.jpg
depth/train/a001_depth.png  错误，主干不一致

depth/train/a001.png
depth/train/a001.jpg      错误，重复主干
```

### 5.2 `load_depth_image`

作用：

- 读取 depth 图。
- 支持 `.npy` 缓存。
- 如果 depth 是单通道，会复制成 3 通道。
- 根据训练尺寸做 resize。

为什么 depth 要变成 3 通道：

当前 depth backbone 复用了类似 RGB backbone 的第一层卷积输入形式，因此 depth 图会被整理成 3 通道输入。

### 5.3 `get_image_and_label`

作用：

- 先调用原始 YOLODataset 的 `get_image_and_label` 获取 RGB 图和标注。
- 再读取对应 depth 图。
- 如果 RGB 和 depth 尺寸不一致，会 resize 到一致。
- 最后把 depth 放入样本字典：

```python
label["depth_img"] = depth_img
label["depth_file"] = self.depth_files[index]
```

后续训练 batch 中就会同时有：

```text
batch["img"]
batch["depth_img"]
```

## 6. RGB-D 同步数据增强

文件：

```text
/workspace/ultralytics-main_for_genye/ultralytics/data/augment.py
```

作用：

原始 Ultralytics 的增强主要处理 RGB 图。本项目增加了对 `depth_img` 的同步处理，保证 RGB 和 depth 不发生空间错位。

重点关注所有包含：

```python
depth_img
```

的代码块。

需要同步处理的增强包括：

- mosaic
- mixup
- copy-paste
- random perspective
- flip up/down
- flip left/right
- letterbox
- resize
- format transform

为什么这部分重要：

如果 RGB 做了翻转、缩放、mosaic，而 depth 没有做同样变换，那么两个模态就会错位，后面的融合模块学到的是错误对应关系。

新同学改增强策略时必须检查：

```text
RGB 怎么变，depth 也要怎么变。
```

## 7. 模型配置

文件：

```text
/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg.yaml
```

作用：

- 定义 YOLO11-seg 主体结构。
- 增加 depth backbone。
- 设置 RGB-D 融合方式。
- 控制 P3 wavelet、Depth FPN 等开关。

重点字段：

```yaml
nc: 80
scale: 's'
end2end: False
reg_max: 1
rgbd_fusion: 'coord_att_v2'
p3_wavelet_guided: True
p3_wavelet_HF: False
p3_wavelet_adaptive: False
p4_wavelet_guided: False
depth_fpn: True
modality_adaptive_gate: False
```

### 7.1 `reg_max: 1`

表示关闭 DFL，直接回归 box。因此训练日志里：

```text
dfl_loss = 0
```

这是正常现象。

### 7.2 `rgbd_fusion`

控制 RGB-D 融合模块。

常见选项：

```text
se
mamba
cross_v1
cross_v2
coord_att
coord_att_v2
cmm
```

当前最终版本使用：

```yaml
rgbd_fusion: 'coord_att_v2'
```

### 7.3 `p3_wavelet_guided`

控制 P3 层是否使用 wavelet 引导。

P3 是高分辨率特征层，对小目标、边缘和细粒度区域更敏感。当前最终版本开启：

```yaml
p3_wavelet_guided: True
```

### 7.4 `depth_fpn`

控制 depth 分支是否使用轻量 FPN。

当前最终版本开启：

```yaml
depth_fpn: True
```

### 7.5 `backbone` 和 `backboneD`

```yaml
backbone:
  ...

backboneD:
  ...
```

其中：

- `backbone` 是 RGB 分支。
- `backboneD` 是 depth 分支。
- 两个分支结构基本对应，方便在 P3/P4/P5 进行同尺度融合。

## 8. RGB-D 前向流程和融合模块

文件：

```text
/workspace/ultralytics-main_for_genye/ultralytics/nn/tasks.py
```

这是本项目最核心的代码文件。

重点阅读：

```python
_predict_once(...)
CoordAttV2
RGBDCoordAttV2
RGBDWaveletGuidedCoordAttV2
DepthLightFPN
parse_model(...)
```

### 8.1 `_predict_once`

作用：

定义 RGB-D 双分支模型的前向过程。

流程可以理解为：

```text
1. depth 输入 xd 先通过 depth backbone。
2. 保存 depth 分支的 P3/P4/P5 特征。
3. RGB 输入 x 通过 RGB backbone。
4. 保存 RGB 分支的 P3/P4/P5 特征。
5. 如果开启 depth_fpn，对 depth 的 P3/P4/P5 做轻量多尺度融合。
6. 在 P3/P4/P5 分别执行 RGB-D fusion。
7. 融合后的特征进入 YOLO11-seg head。
8. 输出检测框和分割 mask。
```

关键融合位置：

```text
P3: backbone stage 4
P4: backbone stage 6
P5: backbone stage 10
```

### 8.2 `CoordAttV2`

作用：

改进版 coordinate attention，用于融合 RGB 和 depth 特征。

输入：

```python
rgb
depth
```

内部逻辑：

```text
cat(rgb, depth)
    -> height/width coordinate pooling
    -> attention weight
    -> channel projection
    -> residual add
```

默认输出：

```python
return rgb + out
```

直观理解：

Depth 提供结构和几何信息，CoordAttV2 学习哪些空间方向和通道更应该增强 RGB 特征。

### 8.3 `RGBDCoordAttV2`

作用：

给 `CoordAttV2` 包一层统一接口，使其可以被 `parse_model` 统一构建。

接口形式：

```python
forward(rgb, depth)
```

### 8.4 `RGBDWaveletGuidedCoordAttV2`

作用：

在 CoordAttV2 基础上加入 wavelet 引导，主要用于 P3 层。

内部会对 depth 特征做 Haar 分解：

```text
LL: 低频结构
LH/HL/HH: 高频边缘
```

然后通过高频边缘生成 gate，对融合特征做增强。

直观理解：

P3 层分辨率高，更适合关注边缘、小目标和局部细节。wavelet 高频部分能帮助模型更关注边界和局部变化。

### 8.5 `DepthLightFPN`

作用：

给 depth 分支增加轻量 top-down FPN，让 depth 的 P3/P4/P5 之间有多尺度上下文。

开启位置：

```yaml
depth_fpn: True
```

直观理解：

Depth 不只是单层几何特征。通过 FPN，高层语义和低层细节能在 depth 分支内部先融合，再与 RGB 融合。

### 8.6 `parse_model`

作用：

根据 YAML 构建模型。

本项目在原始 parse 逻辑后增加了 RGB-D fusion 模块构建：

```text
根据 rgbd_fusion 选择融合模块
根据 p3_wavelet_guided 选择 P3 是否使用 wavelet-guided fusion
根据 depth_fpn 决定是否加入 DepthLightFPN
最后 append P3/P4/P5 三个融合模块
```

这是做消融实验最常改的位置之一。

## 9. 训练和验证预处理

相关文件：

```text
/workspace/ultralytics-main_for_genye/ultralytics/models/yolo/detect/train.py
/workspace/ultralytics-main_for_genye/ultralytics/models/yolo/detect/val.py
/workspace/ultralytics-main_for_genye/ultralytics/models/yolo/segment/val.py
```

重点看 batch 中 `depth_img` 的处理。

训练时主要逻辑：

```python
batch["img"] = batch["img"].float() / 255

if "depth_img" in batch:
    depth = batch["depth_img"].float()
    if depth_dtype == torch.uint8:
        batch["depth_img"] = depth / 255
    else:
        depth_max = depth.amax(...)
        batch["depth_img"] = depth / (depth_max + 1e-7)
```

作用：

- RGB 按普通图像归一化到 0 到 1。
- depth 如果是 uint8，也除以 255。
- depth 如果不是 uint8，则按最大值归一化。

注意：

不同相机或不同 depth 预处理方式会影响模型结果。方案书里也提到 depth 预处理正负偏移会明显影响效果，所以部署时要固定 depth 预处理规范。

## 10. 如何做消融实验

消融实验主要改：

```text
ultralytics/cfg/models/11/yolo11-seg.yaml
```

### 10.1 只用 CoordAttV2

目标：验证 wavelet 和 depth_fpn 是否有贡献。

```yaml
rgbd_fusion: 'coord_att_v2'
p3_wavelet_guided: False
depth_fpn: False
```

对应已有实验：

```text
coord_attv2-s7
LOG/genye11s_just_coord_att_v2.log
```

### 10.2 CoordAttV2 + P3 wavelet

目标：验证 P3 wavelet 是否有效。

```yaml
rgbd_fusion: 'coord_att_v2'
p3_wavelet_guided: True
depth_fpn: False
```

对应已有实验：

```text
coord_attv2-s3
LOG/genye11s.log
```

### 10.3 CoordAttV2 + P3 wavelet + Depth FPN

目标：当前最终版本。

```yaml
rgbd_fusion: 'coord_att_v2'
p3_wavelet_guided: True
depth_fpn: True
```

对应已有实验：

```text
coord_attv2-s8
LOG/genye11s_just_coord_att_v2+p3+dfpn.log
```

### 10.4 换融合模块

例如测试旧版 CoordAtt：

```yaml
rgbd_fusion: 'coord_att'
```

测试 SE-like 轻量融合：

```yaml
rgbd_fusion: 'se'
```

测试 cross attention：

```yaml
rgbd_fusion: 'cross_v2'
```

每次改动后建议同步修改训练 name，避免覆盖或混淆实验目录。

## 11. 三组关键实验对比

| run | 结构 | best Mask mAP50-95 | final best.pt Mask mAP50-95 | final best.pt Box mAP50-95 |
|---|---|---:|---:|---:|
| `s3` | CoordAttV2 + P3 wavelet | 0.79678 | 0.79603 | 0.82316 |
| `s7` | just CoordAttV2 | 0.79749 | 0.79475 | 0.82420 |
| `s8` | CoordAttV2 + P3 wavelet + Depth FPN | 0.80303 | 0.79863 | 0.82339 |

结论：

```text
当前以分割性能为主，推荐使用 s8/best.pt。
如果部署速度压力较大，可以用 s7/best.pt 作为轻量备选。
```

## 12. 新同学常见问题

### 12.1 为什么只看几个文件，不看完整 Ultralytics？

因为大部分代码仍是官方框架，真正需要理解的是 RGB-D 输入、双分支前向、融合模块和同步增强。先看核心改动，后面再补官方框架细节。

### 12.2 为什么普通 `yolo predict` 不能直接当最终推理？

本项目模型是 RGB-D 双输入，除了 RGB 图还需要 depth 图。普通单 RGB 推理可能缺少 `depth_img`，不符合模型设计。

### 12.3 为什么类别 1 指标比类别 0 低？

从当前结果看，类别 1 的 Mask mAP50-95 明显低于类别 0。可能原因包括：

- 类别 1 样本数量少。
- 类别 1 尺度更小。
- 类别 1 遮挡或叠放更多。
- 标注边界质量不稳定。
- 类别 1 和类别 0 外观相似。

后续如果继续提升，优先分析类别 1 的样本数量、尺度分布、漏标和可视化预测结果。

### 12.4 为什么 `dfl_loss=0`？

因为 YAML 中设置：

```yaml
reg_max: 1
```

当前关闭了 DFL，所以这是正常现象。

### 12.5 改代码前最应该注意什么？

三点：

1. 不要破坏 RGB 和 depth 的空间对齐。
2. 不要让训练配置和实验目录混乱。
3. 每次消融只改一个主要变量，否则很难解释结果。

## 13. 建议的学习任务

给新同学可以安排如下任务：

1. 跑通 `s8/best.pt` 的验证。
2. 找到一张 RGB 图和对应 depth 图，确认路径匹配规则。
3. 在 `dataset.py` 打印一次 `batch["img"]` 和 `batch["depth_img"]` 的 shape。
4. 阅读 `_predict_once`，画出 RGB-D 前向流程图。
5. 修改 YAML 关闭 `depth_fpn`，跑一个小 epoch 验证能否启动。
6. 对比 `s7` 和 `s8` 的 `results.csv`，解释为什么最终选择 `s8`。

完成这些任务后，再进一步看部署、加速和破损件检测等后续工作。
