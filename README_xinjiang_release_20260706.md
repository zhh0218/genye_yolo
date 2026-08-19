# 新疆 RGB-D 包裹分割实验交接说明（2026-07-06）

本目录是当前新疆 RGB-D 三分类实例分割实验的可复现交接包，包含最新代码、模型配置、数据集、曝光增强处理脚本、训练脚本、验证脚本，以及 clean / baoguang 两个主要权重。

## 1. 目录位置

```text
/workspace/ultralytics-main_for_genye_release
```

建议师弟直接在该目录下运行。该 release 目录名不包含空格，便于 shell 命令、脚本和 Python 路径正常解析。

## 2. 主要目录结构

```text
/workspace/ultralytics-main_for_genye_release
├── ultralytics/                         # 修改后的 Ultralytics 代码
├── train.py                             # 训练入口
├── configs/                             # 本实验模型配置
├── scripts/                             # 训练、验证、分析脚本
├── Dataset/                             # 数据集与曝光处理脚本
├── YOLOv11-RGB-D-coord_attv2-genye/      # 预训练权重与实验输出
├── README_xinjiang_release.md           # release 简版说明
└── README_xinjiang_release_20260706.md   # 本中文交接说明
```

## 3. 类别定义

本实验是三分类 YOLO instance segmentation：

```text
class 0: 00，单件 / 不被遮挡的快递
class 1: 01，叠件 / 被遮挡的快递
class 2: 10，截断件 / 被边界裁切的快递
```

当前重点关注 class 1，即 `01` 叠件/遮挡件表现。

## 4. 数据集

数据集已经放在 release 目录内部，不依赖 `/workspace/Datasets/...`。

### 4.1 clean 数据集

```text
Dataset/xinjiang_1500
```

对应 YAML：

```text
Dataset/xinjiang_1500/data_3cls.yaml
```

数量：

```text
train: 1200 images / labels / depth
val:    300 images / labels / depth
```

### 4.2 曝光增强数据集

```text
Dataset/xinjiang_1500_baoguang
```

对应 YAML：

```text
Dataset/xinjiang_1500_baoguang/data_3cls.yaml
Dataset/xinjiang_1500_baoguang/data_3cls_val_expaug.yaml
Dataset/xinjiang_1500_baoguang/data_3cls_val_ab.yaml
```

数量：

```text
train:      2214 images / labels / depth
val:         300 images / labels / depth
val_expaug:  247 images / labels / depth
val_ab:      547 images / labels / depth
```

默认训练使用：

```text
train = images/train    # clean 原图 + _realexp 合成曝光图
val   = images/val      # clean 原始验证集
```

`val_expaug` 用于单独评估合成曝光图表现；`val_ab` 是 clean val + 合成曝光 val。

## 5. 模型配置

当前推荐配置：

```text
configs/yolo11-seg-spatialillum-learnblend-auxoff-xinjiang1500.yaml
```

关键开关：

```yaml
rgbd_fusion: "coord_att_v2"
depth_fpn: True
modality_adaptive_gate: True
gate_mode: "spatial_illum"
fusion_learnable_blend: True
fusion_blend_init: 0.5
rgbd_aux_loss: False
rgbd_gate_loss: False
rgbd_depth_aux_loss: False
```

含义简述：

- `modality_adaptive_gate: True`：开启 RGB/depth 自适应门控。
- `gate_mode: 'spatial_illum'`：门控输入额外加入 RGB 特征的局部均值和局部标准差，作为局部光照先验。
- `fusion_learnable_blend: True`：最终融合不再固定为 `rgb + out`，而是学习 `base` 与 `out` 的融合权重。
- `rgbd_aux_loss: False`：当前正式结果关闭额外 gate/depth 辅助监督。

## 6. 主要代码改动位置

关键代码文件：

```text
ultralytics/nn/tasks.py
ultralytics/utils/loss.py
ultralytics/engine/trainer.py
ultralytics/models/yolo/detect/val.py
ultralytics/cfg/default.yaml
train.py
```

### 6.1 spatial_illum 与 learnable blend

文件：

```text
ultralytics/nn/tasks.py
```

核心逻辑：

```python
b = rgb.mean(dim=1, keepdim=True)
mu = F.avg_pool2d(b, 7, 1, 3)
var = F.avg_pool2d(b * b, 7, 1, 3) - mu * mu
illum_prior = torch.cat([mu, torch.sqrt(var + 1e-6)], dim=1)
```

然后：

```python
g_in = torch.cat([x, self._illum_prior(rgb.detach())], dim=1)
alpha = self.mod_gate(g_in)
base = alpha * rgb + (1 - alpha) * depth
```

最后：

```python
blend = torch.softmax(self.blend_logits, dim=0)
return blend[0] * base + blend[1] * out
```

直观含义：

```text
mu    表示局部平均响应，近似局部亮度强度；
sigma 表示局部响应标准差，近似局部纹理/对比度；
alpha 表示空间位置相关的 RGB/depth 融合权重。
```

目标是让模型在过曝、低纹理区域有机会降低 RGB 权重、增强 depth 分支贡献。

### 6.2 best01.pt 与 best_bus1.pt

文件：

```text
ultralytics/engine/trainer.py
ultralytics/models/yolo/detect/val.py
```

训练时除了 `best.pt` 和 `last.pt`，还会保存：

```text
best01.pt      # class 1 表现较优的权重
best_bus1.pt   # business1 hit 表现较优的权重
```

本次主要交接建议先使用 `best.pt`。

## 7. 已包含权重

### 7.1 S8 初始权重

```text
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
```

训练 clean 和 baoguang 模型时均以该权重作为初始化。

### 7.2 clean 模型权重

```text
YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/
└── xinjiang_1500_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/
    ├── weights/best.pt
    ├── weights/best01.pt
    ├── weights/best_bus1.pt
    ├── weights/last.pt
    ├── args.yaml
    ├── args_portable.yaml
    └── results.csv
```

### 7.3 baoguang 模型权重

```text
YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/
└── xinjiang_1500_baoguang_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/
    ├── weights/best.pt
    ├── weights/best01.pt
    ├── weights/best_bus1.pt
    ├── weights/last.pt
    ├── args.yaml
    ├── args_portable.yaml
    └── results.csv
```

说明：

- `args.yaml` 保留原始训练路径，用于历史记录。
- `args_portable.yaml` 已改成 release 内部路径，方便复现。

## 8. 常用命令

进入目录：

```bash
cd /workspace/ultralytics-main_for_genye_release
```

### 8.1 验证已包含权重

```bash
DEVICE=0 BATCH=12 bash scripts/validate_xinjiang1500_release.sh
```

该脚本会验证：

```text
clean best.pt on clean val
baoguang best.pt on clean val
baoguang best.pt on val_expaug
```

### 8.2 重新训练 clean 版本

```bash
DEVICE=0,1,2 BATCH=36 bash scripts/train_xinjiang1500_clean_release.sh
```

### 8.3 重新训练 baoguang 版本

```bash
DEVICE=0,1,2 BATCH=36 bash scripts/train_xinjiang1500_baoguang_release.sh
```

该版本训练集为 clean + 合成曝光图，验证集为 clean val。

### 8.4 重新训练 baoguang + val_ab 版本

```bash
DEVICE=0 BATCH=12 bash scripts/train_xinjiang1500_baoguang_valab_release.sh
```

该版本训练集为 clean + 合成曝光图，验证集为 clean val + 合成曝光 val。

## 9. 曝光增强处理代码

曝光处理工具在：

```text
Dataset/create_xinjiang_realistic_exposure_aug_dataset.py
Dataset/create_xinjiang_mask_exposure_aug_dataset.py
Dataset/README_exposure_processing.md
```

推荐使用：

```text
Dataset/create_xinjiang_realistic_exposure_aug_dataset.py
```

它会读取 YOLO segmentation label mask，优先选择偏白/浅色包裹，并对 RGB 图像中的包裹区域进行过曝合成。depth 和 label 保持不变并复制。

示例命令：

```bash
cd /workspace/ultralytics-main_for_genye_release
python Dataset/create_xinjiang_realistic_exposure_aug_dataset.py \
  --src /workspace/ultralytics-main_for_genye_release/Dataset/xinjiang_1500 \
  --out /workspace/ultralytics-main_for_genye_release/Dataset/xinjiang_1500_baoguang_regenerated \
  --seed 20260702 \
  --overwrite
```

建议不要直接覆盖当前 `xinjiang_1500_baoguang`，先生成到新目录检查数量和 preview。

## 10. 数据与指标口径

当前核心对比是：

```text
clean 模型：
train = xinjiang_1500 train
val   = xinjiang_1500 val

baoguang 模型：
train = xinjiang_1500_baoguang train
val   = xinjiang_1500_baoguang val
```

其中 `xinjiang_1500_baoguang val` 是 clean 原始验证集，不包含 `_realexp`。这样 clean 和 baoguang 的 best.pt 选择口径一致。

额外评估曝光泛化时，使用：

```text
Dataset/xinjiang_1500_baoguang/data_3cls_val_expaug.yaml
```

业务指标参数：

```text
stack_metric_cls = 1
stack_metric_conf = 0.55
business_stack_close_mm = 12.25
business_stack_pix_to_mm = 0.35
```

## 11. 注意事项

1. 不要混用普通 Ultralytics 原版代码加载这些权重，因为权重依赖本目录里的 RGB-D fusion 改动。
2. 如果要复现训练，优先使用 `args_portable.yaml` 或本目录 `scripts/` 下的脚本。
3. 如果显存不足，可以降低 `BATCH`，例如单卡使用 `BATCH=12`。
4. 当前正式结果关闭了 `rgbd_aux_loss`，不要误开辅助 loss 后直接和当前结果比较。
5. `best01.pt` 和 `best_bus1.pt` 是补充权重，默认对外汇报建议使用 `best.pt`。
6. `val_expaug` 是合成曝光验证集，只适合评估曝光增强鲁棒性，不等价于真实现场过曝数据。
7. 本目录还保留了一些历史 wandb/log/runs 文件，实际复现实验主要看 `configs/`、`scripts/`、`Dataset/` 和 `xinjiang_1500_experiments/`。

## 12. 推荐交接方式

如果师弟只需要推理和复现当前实验，重点看：

```text
README_xinjiang_release_20260706.md
configs/yolo11-seg-spatialillum-learnblend-auxoff-xinjiang1500.yaml
scripts/validate_xinjiang1500_release.sh
scripts/train_xinjiang1500_clean_release.sh
scripts/train_xinjiang1500_baoguang_release.sh
Dataset/xinjiang_1500
Dataset/xinjiang_1500_baoguang
YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/*/weights/best.pt
```

如果师弟还要重新生成曝光数据，再看：

```text
Dataset/README_exposure_processing.md
Dataset/create_xinjiang_realistic_exposure_aug_dataset.py
```
