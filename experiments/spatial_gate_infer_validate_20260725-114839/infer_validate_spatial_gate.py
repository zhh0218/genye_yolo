from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "Dataset" / "xinjiang_1500_baoguang"
DEFAULT_MODEL_YAML = ROOT / "configs" / "yolo11-seg-spatialillum-learnblend-auxoff-xinjiang1500.yaml"
DEFAULT_WEIGHTS = (
    ROOT
    / "YOLOv11-RGB-D-coord_attv2-genye"
    / "xinjiang_1500_experiments"
    / "xinjiang_1500_baoguang_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702"
    / "weights"
    / "best.pt"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RGB-D YOLO validation/inference and export spatial gate RGB/depth weights."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="RGB-D dataset root.")
    parser.add_argument("--model-yaml", type=Path, default=DEFAULT_MODEL_YAML, help="Model yaml with spatial gate enabled.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Checkpoint to load.")
    parser.add_argument("--split", default="val_ab", help="Dataset split directory to inspect, e.g. val, val_expaug, val_ab.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference size.")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")
    parser.add_argument("--max-images", type=int, default=50, help="Max images for gate statistics; 0 means all.")
    parser.add_argument("--run-val", action="store_true", help="Also run Ultralytics val on the selected dataset.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory.")
    return parser.parse_args()


def write_windows_data_yaml(dataset: Path, out_dir: Path) -> Path:
    data = {
        "path": dataset.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if (dataset / "images" / "test").exists() else "images/val",
        "depth_train": "depth/train",
        "depth_val": "depth/val",
        "depth_test": "depth/test" if (dataset / "depth" / "test").exists() else "depth/val",
        "nc": 3,
        "names": {0: "00", 1: "01", 2: "10"},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "data_3cls_windows_rgbd.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def image_depth_pairs(dataset: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset / "images" / split
    depth_dir = dataset / "depth" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image split directory: {image_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing depth split directory: {depth_dir}")

    pairs = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        depth_path = depth_dir / f"{image_path.stem}.png"
        if not depth_path.exists():
            matches = list(depth_dir.glob(f"{image_path.stem}.*"))
            depth_path = matches[0] if matches else depth_path
        if depth_path.exists():
            pairs.append((image_path, depth_path))
    if not pairs:
        raise RuntimeError(f"No RGB/depth pairs found under {image_dir} and {depth_dir}")
    return pairs


def sample_group(image_path: Path) -> str:
    return "overexposed" if "_realexp" in image_path.stem else "normal"


def letterbox(im: np.ndarray, size: int = 640, value: int = 114) -> np.ndarray:
    h, w = im.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(value, value, value))


def to_tensor_bchw(im: np.ndarray, device: torch.device) -> torch.Tensor:
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, axis=2)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    # Match Ultralytics predictor convention: BGR image array -> RGB tensor.
    im = im[..., ::-1].transpose(2, 0, 1)
    im = np.ascontiguousarray(im)
    return torch.from_numpy(im).unsqueeze(0).to(device).float() / 255.0


def collect_gate_stats(model: YOLO, image_path: Path, depth_path: Path, imgsz: int, device: torch.device) -> list[dict]:
    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise RuntimeError(f"Failed to read RGB image: {image_path}")
    if depth is None:
        raise RuntimeError(f"Failed to read depth image: {depth_path}")
    if depth.ndim == 2:
        depth = np.repeat(depth[..., None], 3, axis=2)
    elif depth.shape[2] == 1:
        depth = np.repeat(depth, 3, axis=2)
    elif depth.shape[2] > 3:
        depth = depth[:, :, :3]

    rgb_t = to_tensor_bchw(letterbox(rgb, imgsz), device)
    depth_t = to_tensor_bchw(letterbox(depth, imgsz), device)

    raw_model = model.model
    raw_model.eval()
    with torch.inference_mode():
        _ = raw_model(rgb_t, depth_t)

    rows = []
    for module_name, module in raw_model.named_modules():
        alpha = getattr(module, "_last_alpha", None)
        if alpha is None:
            continue
        alpha = alpha.detach().float().cpu()
        base_blend_weight = ""
        out_blend_weight = ""
        blend_logits = getattr(module, "blend_logits", None)
        if blend_logits is not None:
            blend = torch.softmax(blend_logits.detach().float().cpu(), dim=0)
            base_blend_weight = float(blend[0])
            out_blend_weight = float(blend[1])
        rows.append(
            {
                "image": image_path.name,
                "depth": depth_path.name,
                "sample_group": sample_group(image_path),
                "module": module_name,
                "gate_mode": getattr(module, "gate_mode", ""),
                "shape": "x".join(str(x) for x in alpha.shape),
                "src_weight_mean": float(alpha.mean()),
                "src_weight_std": float(alpha.std(unbiased=False)),
                "src_weight_min": float(alpha.min()),
                "src_weight_max": float(alpha.max()),
                "depth_weight_mean": float((1.0 - alpha).mean()),
                "depth_weight_std": float((1.0 - alpha).std(unbiased=False)),
                "depth_weight_min": float((1.0 - alpha).min()),
                "depth_weight_max": float((1.0 - alpha).max()),
                "base_blend_weight": base_blend_weight,
                "out_blend_weight": out_blend_weight,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data_yaml = write_windows_data_yaml(args.dataset, args.out)
    model = YOLO(str(args.model_yaml))
    model.load(str(args.weights))
    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device.split(',')[0]}")
    model.model.to(device)

    summary = {
        "dataset": str(args.dataset),
        "model_yaml": str(args.model_yaml),
        "weights": str(args.weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "device": args.device,
        "data_yaml": str(data_yaml),
    }

    if args.run_val:
        metrics = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, device=args.device, batch=1)
        summary["val_results"] = str(metrics)

    pairs = image_depth_pairs(args.dataset, args.split)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    rows = []
    for index, (image_path, depth_path) in enumerate(pairs, start=1):
        rows.extend(collect_gate_stats(model, image_path, depth_path, args.imgsz, device))
        print(f"[{index}/{len(pairs)}] {image_path.name}")

    csv_path = args.out / "spatial_gate_rgb_depth_weights.csv"
    fieldnames = [
        "image",
        "depth",
        "sample_group",
        "module",
        "gate_mode",
        "shape",
        "src_weight_mean",
        "src_weight_std",
        "src_weight_min",
        "src_weight_max",
        "depth_weight_mean",
        "depth_weight_std",
        "depth_weight_min",
        "depth_weight_max",
        "base_blend_weight",
        "out_blend_weight",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary["num_pairs_checked"] = len(pairs)
    summary["num_gate_rows"] = len(rows)
    summary["group_counts"] = {
        group: sum(1 for image_path, _ in pairs if sample_group(image_path) == group)
        for group in sorted({sample_group(image_path) for image_path, _ in pairs})
    }
    summary["gate_csv"] = str(csv_path)
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not rows:
        print("WARNING: no _last_alpha was captured. Check that the yaml and checkpoint really use modality_adaptive_gate.")
    print(f"Saved gate weights: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
