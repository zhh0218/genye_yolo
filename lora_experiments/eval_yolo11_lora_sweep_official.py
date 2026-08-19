#!/usr/bin/env python3
"""Evaluate YOLOv11s full fine-tune and Conv-LoRA variants on test_fixed.

This script is intended to run on the Linux training server.
It writes compact official Ultralytics metrics for each variant so we do not
need to copy numbers manually from the terminal.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path("/workspace/ultralytics-main_for_genye")
DATA = Path("/workspace/Datasets/genye_ft_splits_seed20260526_v2/yolo11/test_fixed.yaml")
PROJECT = REPO / "YOLOv11-RGB-D-coord_attv2-genye/finetune_genye_lora/eval_test_fixed_lora_sweep_official"
OUT_DIR = REPO / "YOLOv11-RGB-D-coord_attv2-genye/finetune_genye_lora/eval_test_fixed_lora_sweep_summary"

VARIANTS = {
    "full_ft1000": REPO / "YOLOv11-RGB-D-coord_attv2-genye/finetune_genye/genye_ft1000_yolo11s/weights/best.pt",
    "lora_r8": REPO
    / "YOLOv11-RGB-D-coord_attv2-genye/finetune_genye_lora/genye_ft1000_yolo11s_conv_lora_r8/weights/best.pt",
    "lora_r16": REPO
    / "YOLOv11-RGB-D-coord_attv2-genye/finetune_genye_lora/genye_ft1000_yolo11s_conv_lora_r16/weights/best.pt",
    "lora_r32": REPO
    / "YOLOv11-RGB-D-coord_attv2-genye/finetune_genye_lora/genye_ft1000_yolo11s_conv_lora_r32/weights/best.pt",
    "lora_r16_wide_headfusion": REPO
    / "YOLOv11-RGB-D-coord_attv2-genye/finetune_genye_lora/genye_ft1000_yolo11s_conv_lora_r16_wide_headfusion/weights/best.pt",
}


def scalar(x):
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def class_rows(metrics, variant: str, weight: Path, save_dir: Path) -> list[dict]:
    rows: list[dict] = []
    names = getattr(metrics, "names", None) or {0: "0", 1: "1"}
    for cls_idx in sorted(int(k) for k in names):
        try:
            values = metrics.class_result(cls_idx)
        except Exception:
            continue
        values = list(values)
        while len(values) < 8:
            values.append(None)
        rows.append(
            {
                "variant": variant,
                "weight": str(weight),
                "save_dir": str(save_dir),
                "class_id": cls_idx,
                "class_name": str(names.get(cls_idx, cls_idx)),
                "box_p": scalar(values[0]),
                "box_r": scalar(values[1]),
                "box_map50": scalar(values[2]),
                "box_map50_95": scalar(values[3]),
                "mask_p": scalar(values[4]),
                "mask_r": scalar(values[5]),
                "mask_map50": scalar(values[6]),
                "mask_map50_95": scalar(values[7]),
            }
        )
    return rows


def all_row(metrics, variant: str, weight: Path, save_dir: Path) -> dict:
    return {
        "variant": variant,
        "weight": str(weight),
        "save_dir": str(save_dir),
        "class_id": "all",
        "class_name": "all",
        "box_p": scalar(getattr(metrics.box, "mp", None)),
        "box_r": scalar(getattr(metrics.box, "mr", None)),
        "box_map50": scalar(getattr(metrics.box, "map50", None)),
        "box_map50_95": scalar(getattr(metrics.box, "map", None)),
        "mask_p": scalar(getattr(metrics.seg, "mp", None)),
        "mask_r": scalar(getattr(metrics.seg, "mr", None)),
        "mask_map50": scalar(getattr(metrics.seg, "map50", None)),
        "mask_map50_95": scalar(getattr(metrics.seg, "map", None)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    sys.path.insert(0, str(REPO))
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    raw = {}
    for variant, weight in VARIANTS.items():
        if not weight.exists():
            raise FileNotFoundError(weight)
        model = YOLO(str(weight))
        metrics = model.val(
            data=str(DATA),
            imgsz=640,
            batch=16,
            device=3,
            workers=2,
            plots=False,
            save_json=False,
            project=str(PROJECT),
            name=variant,
            exist_ok=True,
            verbose=True,
        )
        save_dir = Path(metrics.save_dir)
        rows = [all_row(metrics, variant, weight, save_dir), *class_rows(metrics, variant, weight, save_dir)]
        all_rows.extend(rows)
        raw[variant] = {
            "weight": str(weight),
            "save_dir": str(save_dir),
            "results_dict": getattr(metrics, "results_dict", {}),
            "rows": rows,
        }
        print(f"OFFICIAL_DONE {variant} {save_dir}")

    write_csv(OUT_DIR / "official_test_fixed_metrics.csv", all_rows)
    (OUT_DIR / "official_test_fixed_metrics.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"OFFICIAL_CSV {OUT_DIR / 'official_test_fixed_metrics.csv'}")


if __name__ == "__main__":
    main()
