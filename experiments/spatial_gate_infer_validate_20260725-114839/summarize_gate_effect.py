from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = Path(__file__).resolve().parent / "outputs" / "spatial_gate_rgb_depth_weights.csv"
DEFAULT_IMAGE_DIR = ROOT / "Dataset" / "xinjiang_1500_baoguang" / "images" / "val_ab"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "spatial_gate_summary_by_group.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize normal vs overexposed spatial gate behavior.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def pstdev(values: list[float]) -> float:
    if not values:
        return float("nan")
    m = mean(values)
    return float(math.sqrt(sum((x - m) ** 2 for x in values) / len(values)))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return float(values[idx])


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def image_light_stats(image_dir: Path, image_name: str) -> dict[str, float]:
    im = cv2.imread(str(image_dir / image_name), cv2.IMREAD_COLOR)
    if im is None:
        return {"brightness_mean": float("nan"), "over088_ratio": float("nan"), "dark012_ratio": float("nan")}
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return {
        "brightness_mean": float(gray.mean()),
        "over088_ratio": float((gray > 0.88).mean()),
        "dark012_ratio": float((gray < 0.12).mean()),
    }


def base_stem(image_name: str) -> str:
    p = Path(image_name)
    return p.stem.replace("_realexp", "")


def summarize_values(rows: list[dict], key: str) -> dict[str, float]:
    values = [float(r[key]) for r in rows]
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values) if values else float("nan"),
        "p10": quantile(values, 0.10),
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "max": max(values) if values else float("nan"),
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8", newline="")))
    for row in rows:
        row.update(image_light_stats(args.image_dir, row["image"]))

    by_module_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_module_group[row["module"]][row["sample_group"]].append(row)

    modules = {}
    for module, groups in by_module_group.items():
        modules[module] = {}
        normal = groups.get("normal", [])
        over = groups.get("overexposed", [])
        for group_name, group_rows in groups.items():
            modules[module][group_name] = {
                "n": len(group_rows),
                "spatial_illum_src_alpha": summarize_values(group_rows, "src_weight_mean"),
                "spatial_illum_depth_1_minus_alpha": summarize_values(group_rows, "depth_weight_mean"),
                "local_src_alpha_min": summarize_values(group_rows, "src_weight_min"),
                "local_src_alpha_max": summarize_values(group_rows, "src_weight_max"),
                "learnable_blend_base": summarize_values(group_rows, "base_blend_weight"),
                "learnable_blend_out": summarize_values(group_rows, "out_blend_weight"),
                "brightness_mean": summarize_values(group_rows, "brightness_mean"),
                "over088_ratio": summarize_values(group_rows, "over088_ratio"),
            }
        if normal and over:
            modules[module]["overexposed_minus_normal"] = {
                "src_alpha_mean_delta": mean([float(r["src_weight_mean"]) for r in over])
                - mean([float(r["src_weight_mean"]) for r in normal]),
                "depth_weight_mean_delta": mean([float(r["depth_weight_mean"]) for r in over])
                - mean([float(r["depth_weight_mean"]) for r in normal]),
                "over088_ratio_delta": mean([float(r["over088_ratio"]) for r in over])
                - mean([float(r["over088_ratio"]) for r in normal]),
            }
        all_rows = normal + over
        modules[module]["correlations"] = {
            "brightness_vs_src_alpha": corr(
                [float(r["brightness_mean"]) for r in all_rows],
                [float(r["src_weight_mean"]) for r in all_rows],
            ),
            "over088_ratio_vs_src_alpha": corr(
                [float(r["over088_ratio"]) for r in all_rows],
                [float(r["src_weight_mean"]) for r in all_rows],
            ),
            "over088_ratio_vs_depth_weight": corr(
                [float(r["over088_ratio"]) for r in all_rows],
                [float(r["depth_weight_mean"]) for r in all_rows],
            ),
        }

    by_pair_module: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_pair_module[base_stem(row["image"])][row["module"]][row["sample_group"]] = row

    paired_deltas = defaultdict(list)
    for stem, module_rows in by_pair_module.items():
        for module, group_rows in module_rows.items():
            normal = group_rows.get("normal")
            over = group_rows.get("overexposed")
            if not normal or not over:
                continue
            paired_deltas[module].append(
                {
                    "base_stem": stem,
                    "src_alpha_delta_over_minus_normal": float(over["src_weight_mean"])
                    - float(normal["src_weight_mean"]),
                    "depth_delta_over_minus_normal": float(over["depth_weight_mean"])
                    - float(normal["depth_weight_mean"]),
                    "over088_delta": float(over["over088_ratio"]) - float(normal["over088_ratio"]),
                    "normal_image": normal["image"],
                    "overexposed_image": over["image"],
                }
            )

    paired_summary = {}
    for module, deltas in paired_deltas.items():
        src_d = [d["src_alpha_delta_over_minus_normal"] for d in deltas]
        depth_d = [d["depth_delta_over_minus_normal"] for d in deltas]
        exp_d = [d["over088_delta"] for d in deltas]
        paired_summary[module] = {
            "n_pairs": len(deltas),
            "src_alpha_delta": {
                "mean": mean(src_d),
                "p10": quantile(src_d, 0.10),
                "p50": quantile(src_d, 0.50),
                "p90": quantile(src_d, 0.90),
                "fraction_src_decreased_on_overexposure": mean([1.0 if x < 0 else 0.0 for x in src_d]),
            },
            "depth_delta": {
                "mean": mean(depth_d),
                "p10": quantile(depth_d, 0.10),
                "p50": quantile(depth_d, 0.50),
                "p90": quantile(depth_d, 0.90),
                "fraction_depth_increased_on_overexposure": mean([1.0 if x > 0 else 0.0 for x in depth_d]),
            },
            "corr_over088_delta_vs_src_delta": corr(exp_d, src_d),
            "largest_depth_increase_examples": sorted(
                deltas, key=lambda d: d["depth_delta_over_minus_normal"], reverse=True
            )[:10],
        }

    result = {
        "csv": str(args.csv),
        "image_dir": str(args.image_dir),
        "total_rows": len(rows),
        "total_images": len({r["image"] for r in rows}),
        "group_counts": {
            group: len({r["image"] for r in rows if r["sample_group"] == group})
            for group in sorted({r["sample_group"] for r in rows})
        },
        "modules": modules,
        "paired_overexposure_summary": paired_summary,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
