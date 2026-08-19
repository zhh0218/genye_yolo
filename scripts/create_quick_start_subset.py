from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path


SRC_ROOT = Path("/workspace/Datasets/final")
OUT_ROOT = Path("/workspace/ultralytics-main_for_genye/examples/quick_start")
SEED = 20260519
TRAIN_COUNT = 80
VAL_COUNT = 20

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def collect_pairs(split: str) -> list[tuple[Path, Path, Path]]:
    image_dir = SRC_ROOT / "images" / split
    depth_dir = SRC_ROOT / "depth" / split
    label_dir = SRC_ROOT / "labels" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    depth_by_stem: dict[str, Path] = {}
    for depth_path in sorted(depth_dir.iterdir()):
        if depth_path.is_file() and depth_path.suffix.lower() in DEPTH_EXTS:
            if depth_path.stem in depth_by_stem:
                raise RuntimeError(f"Duplicate depth stem in {depth_dir}: {depth_path.stem}")
            depth_by_stem[depth_path.stem] = depth_path

    pairs: list[tuple[Path, Path, Path]] = []
    missing_depth = 0
    missing_label = 0
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        depth_path = depth_by_stem.get(image_path.stem)
        label_path = label_dir / f"{image_path.stem}.txt"
        if depth_path is None:
            missing_depth += 1
            continue
        if not label_path.exists():
            missing_label += 1
            continue
        pairs.append((image_path, depth_path, label_path))

    print(
        f"{split}: usable={len(pairs)}, missing_depth={missing_depth}, "
        f"missing_label={missing_label}"
    )
    return pairs


def copy_pairs(split: str, pairs: list[tuple[Path, Path, Path]]) -> None:
    for folder in ("images", "depth", "labels"):
        (OUT_ROOT / "dataset" / folder / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for image_path, depth_path, label_path in pairs:
        dst_image = OUT_ROOT / "dataset" / "images" / split / image_path.name
        dst_depth = OUT_ROOT / "dataset" / "depth" / split / depth_path.name
        dst_label = OUT_ROOT / "dataset" / "labels" / split / label_path.name
        shutil.copy2(image_path, dst_image)
        shutil.copy2(depth_path, dst_depth)
        shutil.copy2(label_path, dst_label)
        manifest_rows.append(
            {
                "split": split,
                "image": str(dst_image),
                "depth": str(dst_depth),
                "label": str(dst_label),
            }
        )
    return manifest_rows


def write_yaml() -> None:
    yaml_text = f"""# Quick-start RGB-D subset for learning and smoke tests.
# RGB paths are explicit; depth paths are inferred by replacing /images/ with /depth/.

train: {OUT_ROOT}/dataset/images/train
val: {OUT_ROOT}/dataset/images/val
test: {OUT_ROOT}/dataset/images/val

nc: 2
names:
  0: 00
  1: 01
"""
    (OUT_ROOT / "quick_start_rgbd.yaml").write_text(yaml_text, encoding="utf-8")


def write_readme() -> None:
    readme = f"""# Quick Start RGB-D Subset

This folder contains a small 100-pair RGB-D subset for new students to run quick smoke tests.

## Contents

```text
examples/quick_start/
├── quick_start_rgbd.yaml
├── manifest.csv
└── dataset/
    ├── images/
    │   ├── train/  # {TRAIN_COUNT} RGB images
    │   └── val/    # {VAL_COUNT} RGB images
    ├── depth/
    │   ├── train/  # {TRAIN_COUNT} depth images
    │   └── val/    # {VAL_COUNT} depth images
    └── labels/
        ├── train/  # {TRAIN_COUNT} YOLO segmentation labels
        └── val/    # {VAL_COUNT} YOLO segmentation labels
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

- This subset is sampled with seed `{SEED}`.
- It is for onboarding, debugging, and smoke testing only.
- For formal experiments, use `/workspace/Datasets/final` and the full validation set.
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)
    train_pairs = collect_pairs("train")
    val_pairs = collect_pairs("val")
    if len(train_pairs) < TRAIN_COUNT:
        raise RuntimeError(f"Not enough train pairs: {len(train_pairs)} < {TRAIN_COUNT}")
    if len(val_pairs) < VAL_COUNT:
        raise RuntimeError(f"Not enough val pairs: {len(val_pairs)} < {VAL_COUNT}")

    sampled_train = rng.sample(train_pairs, TRAIN_COUNT)
    sampled_val = rng.sample(val_pairs, VAL_COUNT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = copy_pairs("train", sampled_train)
    rows.extend(copy_pairs("val", sampled_val))

    manifest_path = OUT_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "depth", "label"])
        writer.writeheader()
        writer.writerows(rows)

    write_yaml()
    write_readme()
    print(f"Created quick-start subset at {OUT_ROOT}")
    print(f"Train pairs: {TRAIN_COUNT}")
    print(f"Val pairs: {VAL_COUNT}")
    print(f"Manifest: {manifest_path}")
    print(f"YAML: {OUT_ROOT / 'quick_start_rgbd.yaml'}")


if __name__ == "__main__":
    main()
