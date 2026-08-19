from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
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
DEFAULT_OUT = EXP_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze spatial gate weights inside/outside local overexposed regions."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-yaml", type=Path, default=DEFAULT_MODEL_YAML)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", default="val_ab")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threshold", type=float, default=0.88, help="Brightness threshold for overexposed mask.")
    parser.add_argument("--min-mask-ratio", type=float, default=0.005, help="Skip tiny masks below this ratio.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def sample_group(image_path: Path) -> str:
    return "overexposed" if "_realexp" in image_path.stem else "normal"


def image_depth_pairs(dataset: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset / "images" / split
    depth_dir = dataset / "depth" / split
    pairs = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        depth_path = depth_dir / f"{image_path.stem}.png"
        if depth_path.exists():
            pairs.append((image_path, depth_path))
    return pairs


def letterbox(im: np.ndarray, size: int = 640, value: int = 114) -> np.ndarray:
    h, w = im.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
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
    im = im[..., ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(im)).unsqueeze(0).to(device).float() / 255.0


def run_one(
    model: YOLO, image_path: Path, depth_path: Path, args: argparse.Namespace, device: torch.device
) -> list[dict]:
    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        return []
    if depth.ndim == 2:
        depth = np.repeat(depth[..., None], 3, axis=2)
    elif depth.ndim == 3 and depth.shape[2] > 3:
        depth = depth[:, :, :3]

    rgb_lb = letterbox(rgb, args.imgsz)
    depth_lb = letterbox(depth, args.imgsz)
    rgb_t = to_tensor_bchw(rgb_lb, device)
    depth_t = to_tensor_bchw(depth_lb, device)

    raw_model = model.model
    raw_model.eval()
    with torch.inference_mode():
        _ = raw_model(rgb_t, depth_t)

    gray = cv2.cvtColor(rgb_lb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    rows = []
    for module_name, module in raw_model.named_modules():
        alpha = getattr(module, "_last_alpha", None)
        if alpha is None:
            continue
        alpha_np = alpha.detach().float().cpu().numpy()[0]
        alpha_2d = alpha_np.mean(axis=0) if alpha_np.ndim == 3 else alpha_np
        ah, aw = alpha_2d.shape
        local_gray = cv2.resize(gray, (aw, ah), interpolation=cv2.INTER_AREA)
        mask = local_gray > args.threshold
        mask_ratio = float(mask.mean())
        if mask_ratio < args.min_mask_ratio or mask_ratio > 0.95:
            continue
        inside_src = float(alpha_2d[mask].mean())
        outside_src = float(alpha_2d[~mask].mean())
        rows.append(
            {
                "image": image_path.name,
                "sample_group": sample_group(image_path),
                "module": module_name,
                "alpha_shape": f"{ah}x{aw}",
                "over_mask_ratio": mask_ratio,
                "inside_over_src_mean": inside_src,
                "outside_src_mean": outside_src,
                "inside_minus_outside_src": inside_src - outside_src,
                "inside_over_depth_mean": 1.0 - inside_src,
                "outside_depth_mean": 1.0 - outside_src,
                "inside_minus_outside_depth": (1.0 - inside_src) - (1.0 - outside_src),
            }
        )
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cpu" if args.device == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device.split(',')[0]}"
    )

    model = YOLO(str(args.model_yaml))
    model.load(str(args.weights))
    model.model.to(device)

    rows = []
    pairs = image_depth_pairs(args.dataset, args.split)
    for index, (image_path, depth_path) in enumerate(pairs, start=1):
        rows.extend(run_one(model, image_path, depth_path, args, device))
        print(f"[{index}/{len(pairs)}] {image_path.name}")

    csv_path = args.out / "local_overexposure_gate_stats.csv"
    fieldnames = [
        "image",
        "sample_group",
        "module",
        "alpha_shape",
        "over_mask_ratio",
        "inside_over_src_mean",
        "outside_src_mean",
        "inside_minus_outside_src",
        "inside_over_depth_mean",
        "outside_depth_mean",
        "inside_minus_outside_depth",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["module"], row["sample_group"])].append(row)
    summary = {}
    for (module, group), group_rows in groups.items():
        key = f"{module}/{group}"
        src_delta = [float(r["inside_minus_outside_src"]) for r in group_rows]
        depth_delta = [float(r["inside_minus_outside_depth"]) for r in group_rows]
        mask_ratio = [float(r["over_mask_ratio"]) for r in group_rows]
        summary[key] = {
            "n": len(group_rows),
            "over_mask_ratio_mean": mean(mask_ratio),
            "inside_minus_outside_src_mean": mean(src_delta),
            "inside_minus_outside_depth_mean": mean(depth_delta),
            "fraction_inside_src_lower": mean([1.0 if x < 0 else 0.0 for x in src_delta]),
            "corr_mask_ratio_vs_inside_src_delta": corr(mask_ratio, src_delta),
        }

    summary_path = args.out / "local_overexposure_gate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved local stats: {csv_path}")
    print(f"Saved local summary: {summary_path}")


if __name__ == "__main__":
    main()
