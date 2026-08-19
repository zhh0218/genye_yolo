# Quick Start RGB-D Subset

This folder contains a small 100-pair RGB-D subset for new students to run quick smoke tests.

## Contents

```text
examples/quick_start/
├── quick_start_rgbd.yaml
├── manifest.csv
└── dataset/
    ├── images/
    │   ├── train/  # 80 RGB images
    │   └── val/    # 20 RGB images
    ├── depth/
    │   ├── train/  # 80 depth images
    │   └── val/    # 20 depth images
    └── labels/
        ├── train/  # 80 YOLO segmentation labels
        └── val/    # 20 YOLO segmentation labels
```

## Important Pairing Rule

Depth images are matched automatically by replacing `/images/` with `/depth/`.
Therefore RGB, depth, and label files must share the same filename stem.

Example:

```text
dataset/images/train/xxx.jpg
dataset/depth/train/xxx.png
dataset/labels/train/xxx.txt
```

## Quick Validation

```bash
cd /workspace/ultralytics-main_for_genye
conda activate yolo26
python - <<'PY'
from ultralytics import YOLO

model = YOLO("/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt")
model.val(
    data="/workspace/ultralytics-main_for_genye/examples/quick_start/quick_start_rgbd.yaml",
    imgsz=640,
    batch=8,
    device="0",
    split="val",
)
PY
```

## Quick Training Smoke Test

This is only for checking that the training pipeline works. Do not report these numbers as formal results.

```bash
cd /workspace/ultralytics-main_for_genye
conda activate yolo26
python - <<'PY'
from ultralytics import YOLO

model = YOLO("/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg.yaml")
model.load("/workspace/ultralytics-main_for_genye/yolo11l.pt")
model.train(
    data="/workspace/ultralytics-main_for_genye/examples/quick_start/quick_start_rgbd.yaml",
    imgsz=640,
    epochs=2,
    batch=4,
    workers=2,
    device="0",
    optimizer="MuSGD",
    amp=True,
    project="examples/quick_start/runs",
    name="smoke_train",
)
PY
```

## Notes

- This subset is sampled with seed `20260519`.
- It is for onboarding, debugging, and smoke testing only.
- For formal experiments, use `/workspace/Datasets/final` and the full validation set.
