# RGB-D S8 交接包说明

本文档说明如何生成一个干净的交接目录，避免把完整实验环境中的缓存、旧实验、wandb 记录等无关内容一起交给新同学。

## 1. 交接包目标

交接包目录建议命名为：

```text
release_rgbd_s8/
```

它只保留学习和复现最终版本所需的内容：

```text
release_rgbd_s8/
├── README_RGBD_S8.md
├── docs/
├── examples/quick_start/
├── scripts/
├── train_genye.py
├── data-ubuntu-genye.yaml
├── yolo11l.pt
├── ultralytics/
├── LOG/
│   └── genye11s_just_coord_att_v2+p3+dfpn.log
└── YOLOv11-RGB-D-coord_attv2-genye/
    └── coord_attv2-s8/
        ├── args.yaml
        ├── results.csv
        ├── results.png
        ├── confusion_matrix.png
        ├── confusion_matrix_normalized.png
        ├── val_batch*_pred.jpg
        ├── val_batch*_labels.jpg
        └── weights/
            └── best.pt
```

这样 `examples/quick_start/quick_val.py` 和 `quick_train.py` 在交接包中仍然可以按相对路径运行。

## 2. 不建议放进交接包的内容

以下内容建议归档或删除，不放进学习版交接包：

```text
wandb/
__pycache__/
*.pyc
*.cache
早期中断实验目录
重复权重
大批量旧日志
无关官方 examples
```

原因是这些文件会干扰新同学判断“哪个版本才是最终版”。

## 3. 生成交接包

在服务器项目根目录运行：

```bash
cd /workspace/ultralytics-main_for_genye
python scripts/create_release_package.py
```

生成目录：

```text
/workspace/ultralytics-main_for_genye/release_rgbd_s8
```

如果目录已存在，脚本会拒绝覆盖。确认要重新生成时，先手动改名或删除旧目录。

## 4. 交接包验证

生成后进入交接包：

```bash
cd /workspace/ultralytics-main_for_genye/release_rgbd_s8
conda activate yolo26
python examples/quick_start/quick_val.py
```

期望结果：

```text
20 images
0 corrupt
Results saved to examples/quick_start/runs/smoke_val
```

如果报 `DepthLightFPN` 找不到，说明没有正确导入交接包里的本地 `ultralytics/`。

## 5. 正式指标说明

交接包中的 `examples/quick_start/` 只有 100 对样本，只用于 smoke test。

正式指标仍以完整验证集为准：

```text
/workspace/Datasets/final/images/val
```

当前最终权重：

```text
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
```

正式参考指标：

```text
all Mask mAP50-95 = 0.79863
类别1 Mask mAP50-95 = 0.66651
```

## 6. 发给新同学前检查

交接包生成后，至少检查：

```text
[ ] README_RGBD_S8.md 存在
[ ] docs/code_guide.md 存在
[ ] docs/experiment_summary.md 存在
[ ] docs/handover_checklist.md 存在
[ ] examples/quick_start/quick_val.py 能跑通
[ ] examples/quick_start/quick_train.py 能启动
[ ] 最终权重 s8/best.pt 存在
[ ] yolo11l.pt 存在
[ ] ultralytics/ 是本项目修改版
[ ] release_manifest.csv 已生成
```
