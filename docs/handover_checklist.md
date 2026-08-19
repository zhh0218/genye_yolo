# RGB-D YOLO11-seg 项目交接清单

本文档用于项目交接时逐项检查，确保新同学拿到代码后能跑通、能理解、能复现，并且不会混淆最终版本。

## 1. 最终版本确认

当前正式推荐版本：

```text
YOLO11-seg-RGBD-CoordAttV2-P3Wavelet-DepthFPN
```

对应权重：

```text
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
```

对应运行目录：

```text
/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8
```

对应日志：

```text
/workspace/ultralytics-main_for_genye/LOG/genye11s_just_coord_att_v2+p3+dfpn.log
```

交接时必须明确：

```text
最终版 = s8/best.pt
轻量备选 = s7/best.pt
旧主版本 = s3/best.pt
```

不要混用不同实验目录下的 `best.pt`。

## 2. 必须保留的文件

交接代码至少应包含：

```text
README_RGBD_S8.md
docs/code_guide.md
docs/experiment_summary.md
docs/handover_checklist.md
examples/quick_start/
train_genye.py
data-ubuntu-genye.yaml
ultralytics/
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/args.yaml
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/results.csv
LOG/genye11s_just_coord_att_v2+p3+dfpn.log
```

如果空间允许，建议同时保留：

```text
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/results.png
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/confusion_matrix.png
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/confusion_matrix_normalized.png
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/val_batch*_pred.jpg
YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/val_batch*_labels.jpg
```

## 3. 可以不交接或归档的内容

以下内容不建议直接给新同学作为主学习材料，避免目录太乱：

```text
wandb/
__pycache__/
*.cache
早期中断实验目录
重复权重
无关官方 examples
临时 nohup 日志
调试产生的 runtime_cfg
```

如果需要保留历史实验，可以单独放到：

```text
archive/
```

并写清楚这些不是最终版本。

## 4. 环境检查

进入项目目录：

```bash
cd /workspace/ultralytics-main_for_genye
conda activate yolo26
```

检查 Python 和 PyTorch：

```bash
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
```

期望环境：

```text
Python 3.10
PyTorch 2.5.1 + CUDA 12.1
Ultralytics 8.3.195
```

具体小版本可以略有差异，但必须能导入项目本地 `ultralytics/`。

## 5. 本地 ultralytics 导入检查

本项目修改了 `ultralytics/` 源码，不能误用 pip 安装的官方版本。

检查命令：

```bash
cd /workspace/ultralytics-main_for_genye
python - << 'PY'
import ultralytics
import ultralytics.nn.tasks as tasks
print("ultralytics path:", ultralytics.__file__)
print("tasks path:", tasks.__file__)
print("has DepthLightFPN:", hasattr(tasks, "DepthLightFPN"))
print("has RGBDCoordAttV2:", hasattr(tasks, "RGBDCoordAttV2"))
PY
```

正确结果应该类似：

```text
ultralytics path: /workspace/ultralytics-main_for_genye/ultralytics/__init__.py
tasks path: /workspace/ultralytics-main_for_genye/ultralytics/nn/tasks.py
has DepthLightFPN: True
has RGBDCoordAttV2: True
```

如果 `tasks path` 指向：

```text
/root/miniconda3/envs/.../site-packages/ultralytics/
```

说明导入错了，需要从项目根目录运行，或在脚本中加入：

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
```

## 6. quick_start 小数据检查

quick_start 路径：

```text
/workspace/ultralytics-main_for_genye/examples/quick_start
```

应包含：

```text
quick_start_rgbd.yaml
manifest.csv
quick_val.py
quick_train.py
dataset/images/train
dataset/images/val
dataset/depth/train
dataset/depth/val
dataset/labels/train
dataset/labels/val
```

数量检查：

```bash
ROOT=/workspace/ultralytics-main_for_genye/examples/quick_start
find $ROOT/dataset/images/train -maxdepth 1 -type f | wc -l
find $ROOT/dataset/depth/train -maxdepth 1 -type f | wc -l
find $ROOT/dataset/labels/train -maxdepth 1 -type f | wc -l
find $ROOT/dataset/images/val -maxdepth 1 -type f | wc -l
find $ROOT/dataset/depth/val -maxdepth 1 -type f | wc -l
find $ROOT/dataset/labels/val -maxdepth 1 -type f | wc -l
```

期望结果：

```text
80
80
80
20
20
20
```

## 7. 快速验证检查

运行：

```bash
cd /workspace/ultralytics-main_for_genye
conda activate yolo26
python examples/quick_start/quick_val.py
```

期望：

```text
20 images
0 corrupt
Results saved to examples/quick_start/runs/smoke_val
```

当前已知 smoke test 结果约为：

```text
Mask mAP50-95 = 0.94384
```

注意：这个结果只说明流程能跑通，不能作为正式实验指标。

## 8. 快速训练检查

运行：

```bash
cd /workspace/ultralytics-main_for_genye
conda activate yolo26
python examples/quick_start/quick_train.py
```

期望：

```text
训练能正常开始
能读取 RGB 和 depth
能完成 2 epoch
结果保存到 examples/quick_start/runs/smoke_train
```

该训练只用于 smoke test，不用于论文或正式报告。

## 9. 正式验证检查

正式验证必须使用完整验证集：

```text
/workspace/Datasets/final/images/val
```

数据配置：

```text
/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml
```

验证命令：

```bash
cd /workspace/ultralytics-main_for_genye
conda activate yolo26
python - << 'PY'
from ultralytics import YOLO

model = YOLO("/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt")
model.val(
    data="/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml",
    imgsz=640,
    batch=36,
    device="0,1,2,3",
    split="val",
    project="handover_val",
    name="s8_full_val",
    exist_ok=True,
)
PY
```

正式参考指标：

```text
all Box mAP50-95  = 0.82339
all Mask mAP50    = 0.91975
all Mask mAP50-95 = 0.79863
```

类别 1 参考指标：

```text
类别1 Mask mAP50    = 0.85647
类别1 Mask mAP50-95 = 0.66651
```

## 10. 数据路径检查

当前数据配置：

```text
/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml
```

核心内容：

```yaml
train: /workspace/Datasets/final/images/train
val: /workspace/Datasets/final/images/val
test: /workspace/Datasets/final/images/test

nc: 2
names:
    0: 00
    1: 01
```

注意：depth 路径没有显式写在 YAML 中，而是自动推断：

```text
/images/ -> /depth/
```

因此正式数据必须满足：

```text
images/train/xxx.jpg
depth/train/xxx.png
labels/train/xxx.txt
```

三者文件名主干一致。

## 11. 核心代码检查

新同学至少要知道以下文件作用：

| 文件                                        | 作用                                       |
| ------------------------------------------- | ------------------------------------------ |
| `train_genye.py`                            | 训练入口、超参数、权重加载                 |
| `data-ubuntu-genye.yaml`                    | 数据路径和类别定义                         |
| `ultralytics/cfg/models/11/yolo11-seg.yaml` | 模型结构和融合开关                         |
| `ultralytics/nn/tasks.py`                   | RGB-D 前向、CoordAttV2、wavelet、Depth FPN |
| `ultralytics/data/build.py`                 | 自动选择 `YOLORGBDDataset`                 |
| `ultralytics/data/dataset.py`               | RGB-depth 配对和读取                       |
| `ultralytics/data/augment.py`               | RGB-depth 同步增强                         |
| `ultralytics/models/yolo/detect/train.py`   | 训练时 depth 归一化                        |
| `ultralytics/models/yolo/detect/val.py`     | 验证时 depth 归一化                        |

详细说明见：

```text
docs/code_guide.md
```

## 12. 常见交接问题

### 12.1 加载 best.pt 报 `DepthLightFPN` 找不到

原因：导入了 pip 安装的官方 Ultralytics，而不是项目本地修改版。

解决：

```bash
cd /workspace/ultralytics-main_for_genye
python examples/quick_start/quick_val.py
```

或者在脚本开头加入项目根目录到 `sys.path`。

### 12.2 depth 找不到或样本被跳过

检查：

```text
RGB 文件名主干
depth 文件名主干
label 文件名主干
```

三者必须一致。

### 12.3 quick_start 指标很高，能否写进论文或报告？

不能。quick_start 只有 20 张 val，只用于流程检查。正式指标必须使用完整 val。

### 12.4 用 last.pt 还是 best.pt？

交付和正式验证统一使用：

```text
weights/best.pt
```

### 12.5 类别 1 指标低怎么办？

后续优化优先做：

- 类别 1 样本统计。
- 类别 1 尺度分布分析。
- 类别 1 漏标和边界质量检查。
- 类别 1 针对性增强或重采样。
- 类别 1 阈值单独调整。

## 13. 交接通过标准

满足以下条件，可以认为项目交接基本通过：

```text
[ ] 能从项目根目录导入本地 ultralytics。
[ ] `DepthLightFPN=True`，`RGBDCoordAttV2=True`。
[ ] `quick_start` 六个目录数量分别为 80/80/80/20/20/20。
[ ] `python examples/quick_start/quick_val.py` 能跑通。
[ ] `python examples/quick_start/quick_train.py` 能跑通。
[ ] 知道最终权重是 `s8/best.pt`。
[ ] 知道正式指标来自完整 val，不来自 quick_start。
[ ] 知道 RGB-depth 文件名主干必须一致。
[ ] 能说清楚 `s3/s7/s8` 三个实验的区别。
```

如果以上都通过，新同学就可以开始阅读源码和做后续实验。
