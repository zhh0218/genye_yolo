# 新疆 RGB-D 实验 Release 简版说明

本目录是新疆 RGB-D 三分类包裹实例分割实验的 release 包，包含最新代码、模型配置、数据集、曝光增强数据、训练脚本、验证脚本以及 clean / baoguang 两个主要模型权重。

完整中文交接说明请优先阅读：

```text
README_xinjiang_release_20260706.md
```

曝光增强处理代码说明请阅读：

```text
Dataset/README_exposure_processing.md
```

## 1. 目录位置

```text
/workspace/ultralytics-main_for_genye_release
```

建议直接在该目录下运行脚本。

## 2. 核心内容

```text
ultralytics/                         # 修改后的 Ultralytics 代码
train.py                             # 训练入口
configs/                             # 模型配置
scripts/                             # 训练、验证、分析脚本
Dataset/                             # clean / baoguang 数据集与曝光处理脚本
YOLOv11-RGB-D-coord_attv2-genye/      # S8 初始权重与实验输出
README_xinjiang_release_20260706.md   # 完整中文交接说明
```

## 3. 类别定义

```text
class 0: 00，单件 / 不被遮挡的快递
class 1: 01，叠件 / 被遮挡的快递
class 2: 10，截断件 / 被边界裁切的快递
```

## 4. 数据集

clean 数据集：

```text
Dataset/xinjiang_1500/data_3cls.yaml
```

数量：

```text
train: 1200
val:   300
```

曝光增强数据集：

```text
Dataset/xinjiang_1500_baoguang/data_3cls.yaml
```

数量：

```text
train:      2214
val:         300
val_expaug:  247
val_ab:      547
```

说明：

```text
data_3cls.yaml             # train = clean + _realexp，val = clean val
data_3cls_val_expaug.yaml  # val = 合成曝光验证集
data_3cls_val_ab.yaml      # val = clean val + 合成曝光 val
```

## 5. 模型配置

推荐使用：

```text
configs/yolo11-seg-spatialillum-learnblend-auxoff-xinjiang1500.yaml
```

关键开关：

```yaml
rgbd_fusion: 'coord_att_v2'
depth_fpn: True
modality_adaptive_gate: True
gate_mode: 'spatial_illum'
fusion_learnable_blend: True
fusion_blend_init: 0.5
rgbd_aux_loss: False
```

简要理解：

```text
spatial_illum:
  在 RGB-D 融合前，从 RGB feature 中计算局部均值 mu 和局部标准差 sigma，
  作为光照先验输入 gate，用于预测空间 alpha。

alpha:
  控制每个位置 RGB / depth 的融合比例。

fusion_learnable_blend:
  学习 base 融合特征与 CoordAttV2 out 特征的最终融合权重。
```

## 6. 已包含主要权重

S8 初始权重：

```text
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
```

clean 模型：

```text
YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/
xinjiang_1500_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt
```

baoguang 模型：

```text
YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/
xinjiang_1500_baoguang_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt
```

对应实验目录中还保留：

```text
best01.pt
best_bus1.pt
last.pt
args.yaml
args_portable.yaml
results.csv
```

默认对外使用 `best.pt`。

## 7. 常用命令

进入目录：

```bash
cd /workspace/ultralytics-main_for_genye_release
```

验证已有权重：

```bash
DEVICE=0 BATCH=12 bash scripts/validate_xinjiang1500_release.sh
```

重新训练 clean：

```bash
DEVICE=0,1,2 BATCH=36 bash scripts/train_xinjiang1500_clean_release.sh
```

重新训练 baoguang：

```bash
DEVICE=0,1,2 BATCH=36 bash scripts/train_xinjiang1500_baoguang_release.sh
```

重新训练 baoguang + val_ab：

```bash
DEVICE=0 BATCH=12 bash scripts/train_xinjiang1500_baoguang_valab_release.sh
```

## 8. 曝光增强处理

曝光处理脚本：

```text
Dataset/create_xinjiang_realistic_exposure_aug_dataset.py
Dataset/create_xinjiang_mask_exposure_aug_dataset.py
```

推荐使用：

```text
Dataset/create_xinjiang_realistic_exposure_aug_dataset.py
```

说明文档：

```text
Dataset/README_exposure_processing.md
```

## 9. 注意事项

1. 不要用原版 Ultralytics 直接加载本目录权重，因为模型结构包含 RGB-D fusion 改动。
2. 训练和验证优先使用 `scripts/` 下的 release 脚本。
3. `args.yaml` 保留历史绝对路径；`args_portable.yaml` 已改成 release 内部路径。
4. 当前正式结果关闭 `rgbd_aux_loss`，不要误开后直接对比指标。
5. `val_expaug` 是合成曝光验证集，用于评估曝光鲁棒性，不等价于真实现场过曝数据。
6. 更完整说明见 `README_xinjiang_release_20260706.md`。
