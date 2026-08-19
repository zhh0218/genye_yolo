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
        description="Measure spatial gate changes on the exact labeled package instances selected for exposure augmentation."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-yaml", type=Path, default=DEFAULT_MODEL_YAML)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", default="val_ab")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-pairs", type=int, default=0, help="Maximum normal/realexp pairs; 0 means all.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def find_by_suffix(directory: Path, suffix: str) -> Path | None:
    matches = sorted(p for p in directory.glob(f"*{suffix}") if p.is_file())
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous suffix {suffix!r} under {directory}: {matches}")
    return matches[0] if matches else None


def load_selected_instances(manifest_path: Path, split: str) -> dict[str, set[int]]:
    selected: dict[str, set[int]] = defaultdict(set)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split and int(row["selected"]) == 1:
                selected[row["stem"]].add(int(row["instance_index"]))
    return dict(selected)


def read_label_masks(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    masks = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        values = [float(x) for x in parts[1:]]
        if len(values) == 4:
            xc, yc, bw, bh = values
            points = np.array(
                [
                    [(xc - bw / 2) * width, (yc - bh / 2) * height],
                    [(xc + bw / 2) * width, (yc - bh / 2) * height],
                    [(xc + bw / 2) * width, (yc + bh / 2) * height],
                    [(xc - bw / 2) * width, (yc + bh / 2) * height],
                ],
                dtype=np.float32,
            )
        else:
            points = np.asarray(values[: len(values) // 2 * 2], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= width
            points[:, 1] *= height
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) >= 3:
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        masks.append(mask)
    return masks


def letterbox(im: np.ndarray, size: int, interpolation: int, pad_value: int | tuple[int, int, int]) -> np.ndarray:
    height, width = im.shape[:2]
    scale = min(size / height, size / width)
    new_height, new_width = int(round(height * scale)), int(round(width * scale))
    resized = cv2.resize(im, (new_width, new_height), interpolation=interpolation)
    top = (size - new_height) // 2
    bottom = size - new_height - top
    left = (size - new_width) // 2
    right = size - new_width - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)


def image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb_chw = image[..., ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(rgb_chw)).unsqueeze(0).to(device).float() / 255.0


def capture_alpha(
    model: YOLO, rgb_path: Path, depth_path: Path, imgsz: int, device: torch.device
) -> dict[str, np.ndarray]:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise RuntimeError(f"Failed to read RGB-D pair: {rgb_path}, {depth_path}")
    if depth.ndim == 2:
        depth = np.repeat(depth[..., None], 3, axis=2)
    elif depth.ndim == 3 and depth.shape[2] == 1:
        depth = np.repeat(depth, 3, axis=2)
    elif depth.ndim == 3 and depth.shape[2] > 3:
        depth = depth[:, :, :3]

    rgb_lb = letterbox(rgb, imgsz, cv2.INTER_LINEAR, (114, 114, 114))
    depth_lb = letterbox(depth, imgsz, cv2.INTER_LINEAR, (114, 114, 114))
    with torch.inference_mode():
        _ = model.model(image_tensor(rgb_lb, device), image_tensor(depth_lb, device))

    alpha_maps = {}
    for name, module in model.model.named_modules():
        alpha = getattr(module, "_last_alpha", None)
        if alpha is None:
            continue
        array = alpha.detach().float().cpu().numpy()[0]
        alpha_maps[name] = array.mean(axis=0) if array.ndim == 3 else array
    return alpha_maps


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(weights.sum())
    return float((values * weights).sum() / weight_sum) if weight_sum > 1e-8 else float("nan")


def region_means(alpha: np.ndarray, selected_mask: np.ndarray, control_mask: np.ndarray, package_mask: np.ndarray) -> dict[str, float]:
    height, width = alpha.shape
    selected = cv2.resize(selected_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    control = cv2.resize(control_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    packages = cv2.resize(package_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    background = np.clip(1.0 - packages, 0.0, 1.0)
    return {
        "selected_src": weighted_mean(alpha, selected),
        "control_src": weighted_mean(alpha, control),
        "background_src": weighted_mean(alpha, background),
        "selected_feature_area": float(selected.sum()),
        "control_feature_area": float(control.sum()),
    }


def finite(values: list[float]) -> list[float]:
    return [x for x in values if math.isfinite(x)]


def stats(values: list[float]) -> dict[str, float | int]:
    values = finite(values)
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    se = sd / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "n": len(array),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "fraction_positive": float((array > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cpu" if args.device == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device.split(',')[0]}"
    )

    manifest = args.dataset / "meta" / "candidate_instance_manifest.csv"
    selected_by_stem = load_selected_instances(manifest, "val")
    stems = sorted(selected_by_stem)
    if args.max_pairs > 0:
        stems = stems[: args.max_pairs]

    model = YOLO(str(args.model_yaml))
    model.load(str(args.weights))
    model.model.to(device).eval()

    image_dir = args.dataset / "images" / args.split
    depth_dir = args.dataset / "depth" / args.split
    label_dir = args.dataset / "labels" / args.split
    rows = []

    for pair_index, stem in enumerate(stems, start=1):
        normal_image = find_by_suffix(image_dir, f"{stem}.jpg")
        exposed_image = find_by_suffix(image_dir, f"{stem}_realexp.jpg")
        normal_depth = find_by_suffix(depth_dir, f"{stem}.png")
        exposed_depth = find_by_suffix(depth_dir, f"{stem}_realexp.png")
        label_path = find_by_suffix(label_dir, f"{stem}.txt")
        if not all((normal_image, exposed_image, normal_depth, exposed_depth, label_path)):
            raise FileNotFoundError(f"Incomplete pair for manifest stem {stem}")

        original = cv2.imread(str(normal_image), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"Failed to read {normal_image}")
        masks = read_label_masks(label_path, original.shape[1], original.shape[0])
        selected_indices = selected_by_stem[stem]
        if max(selected_indices) >= len(masks):
            raise RuntimeError(f"Manifest instance index exceeds labels for {stem}: {selected_indices}, labels={len(masks)}")

        selected_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        control_mask = np.zeros_like(selected_mask)
        package_mask = np.zeros_like(selected_mask)
        for index, mask in enumerate(masks):
            package_mask |= mask
            if index in selected_indices:
                selected_mask |= mask
            else:
                control_mask |= mask

        selected_lb = letterbox(selected_mask, args.imgsz, cv2.INTER_NEAREST, 0)
        control_lb = letterbox(control_mask, args.imgsz, cv2.INTER_NEAREST, 0)
        package_lb = letterbox(package_mask, args.imgsz, cv2.INTER_NEAREST, 0)
        normal_maps = capture_alpha(model, normal_image, normal_depth, args.imgsz, device)
        exposed_maps = capture_alpha(model, exposed_image, exposed_depth, args.imgsz, device)

        for module in sorted(normal_maps):
            normal_regions = region_means(normal_maps[module], selected_lb, control_lb, package_lb)
            exposed_regions = region_means(exposed_maps[module], selected_lb, control_lb, package_lb)
            selected_delta = exposed_regions["selected_src"] - normal_regions["selected_src"]
            control_delta = exposed_regions["control_src"] - normal_regions["control_src"]
            background_delta = exposed_regions["background_src"] - normal_regions["background_src"]
            rows.append(
                {
                    "stem": stem,
                    "module": module,
                    "selected_instances": len(selected_indices),
                    "all_instances": len(masks),
                    "normal_selected_src": normal_regions["selected_src"],
                    "exposed_selected_src": exposed_regions["selected_src"],
                    "selected_src_delta_exposed_minus_normal": selected_delta,
                    "selected_depth_delta_exposed_minus_normal": -selected_delta,
                    "control_src_delta_exposed_minus_normal": control_delta,
                    "background_src_delta_exposed_minus_normal": background_delta,
                    "selected_vs_control_did_src": selected_delta - control_delta,
                    "selected_vs_background_did_src": selected_delta - background_delta,
                    "normal_selected_minus_background_src": (
                        normal_regions["selected_src"] - normal_regions["background_src"]
                    ),
                    "exposed_selected_minus_background_src": (
                        exposed_regions["selected_src"] - exposed_regions["background_src"]
                    ),
                    "selected_feature_area": normal_regions["selected_feature_area"],
                    "control_feature_area": normal_regions["control_feature_area"],
                }
            )
        print(f"[{pair_index}/{len(stems)}] {stem}")

    csv_path = args.out / "labeled_exposure_gate_stats.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["module"]].append(row)
    summary = {
        "method": {
            "exposed_region_truth": "candidate_instance_manifest selected=1 intersected with the corresponding YOLO label polygon",
            "comparison": "paired normal vs _realexp on the identical selected package mask",
            "controls": "unselected labeled package masks and non-package background",
            "aggregation_unit": "one union mask per image to avoid treating multiple instances from one image as independent samples",
        },
        "pairs": len(stems),
        "selected_instances": sum(len(selected_by_stem[stem]) for stem in stems),
        "modules": {},
    }
    metrics = [
        "selected_src_delta_exposed_minus_normal",
        "selected_depth_delta_exposed_minus_normal",
        "control_src_delta_exposed_minus_normal",
        "background_src_delta_exposed_minus_normal",
        "selected_vs_control_did_src",
        "selected_vs_background_did_src",
        "normal_selected_minus_background_src",
        "exposed_selected_minus_background_src",
    ]
    for module, module_rows in grouped.items():
        summary["modules"][module] = {
            metric: stats([float(row[metric]) for row in module_rows]) for metric in metrics
        }

    summary_path = args.out / "labeled_exposure_gate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
