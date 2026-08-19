from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/ultralytics-main_for_genye_release")
from ultralytics import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator

DATA = "/workspace/ultralytics-main_for_genye_release/Dataset/xinjiang_1500_baoguang/data_3cls_val_expaug.yaml"
OUT_DIR = Path("/workspace/genye_rgbd_route_probe/alpha_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ALPHA_CSV = OUT_DIR / "compare_clean_baoguang_on_xinjiang1500_baoguang_val_expaug_alpha.csv"
OUT_BLEND_CSV = OUT_DIR / "compare_clean_baoguang_on_xinjiang1500_baoguang_val_expaug_blend.csv"
OUT_JSON = OUT_DIR / "compare_clean_baoguang_on_xinjiang1500_baoguang_val_expaug_alpha_blend.json"

RUNS = [
    {
        "model": "clean_xinjiang_1500",
        "weights": "/workspace/ultralytics-main_for_genye_release/YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/xinjiang_1500_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt",
    },
    {
        "model": "baoguang_xinjiang_1500_baoguang",
        "weights": "/workspace/ultralytics-main_for_genye_release/YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/xinjiang_1500_baoguang_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt",
    },
]

STAGE_MAP = {
    "model.model.36.coord_att": "P3/8",
    "model.model.37.coord_att": "P4/16",
    "model.model.38.coord_att": "P5/32",
}
CATEGORIES = [
    "all_pixels",
    "gt_package",
    "gt_pkg_bright088",
    "gt_pkg_over_b088_c004",
]


def append_vals(store, stage, category, alpha, mask=None):
    if mask is None:
        vals = alpha.reshape(-1).detach().float().cpu()
    else:
        m = mask.bool()
        if not bool(m.any()):
            return
        vals = alpha[m].reshape(-1).detach().float().cpu()
    store.setdefault(stage, {}).setdefault(category, []).append(vals)


def make_validator(model_name, blend_rows, alpha_rows_holder):
    class AlphaBlendValidator(SegmentationValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.alpha_modules = []
            for name, module in model.named_modules():
                if type(module).__name__ == "CoordAttV2" and getattr(module, "adaptive_gate", False):
                    self.alpha_modules.append((name, module))
                    blend = [float("nan"), float("nan")]
                    if hasattr(module, "blend_logits"):
                        blend = torch.softmax(module.blend_logits.detach().cpu(), dim=0).tolist()
                    blend_rows.append({
                        "model": model_name,
                        "module": name,
                        "stage": STAGE_MAP.get(name, name),
                        "blend_base": blend[0],
                        "blend_out": blend[1],
                    })
            self.values = {}
            self.pixel_counts = {}

        def update_metrics(self, preds, batch):
            img = batch["img"].detach().float()
            pkg_mask = (batch["masks"].detach().float() > 0).unsqueeze(1)

            bright = img.mean(dim=1, keepdim=True)
            mu = F.avg_pool2d(bright, 7, 1, 3)
            var = (F.avg_pool2d(bright * bright, 7, 1, 3) - mu * mu).clamp_min(0)
            contrast = torch.sqrt(var + 1e-6)

            bright_m = F.interpolate(bright, size=pkg_mask.shape[-2:], mode="bilinear", align_corners=False)
            contrast_m = F.interpolate(contrast, size=pkg_mask.shape[-2:], mode="bilinear", align_corners=False)
            masks_m = {
                "gt_package": pkg_mask,
                "gt_pkg_bright088": pkg_mask & (bright_m > 0.88),
                "gt_pkg_over_b088_c004": pkg_mask & (bright_m > 0.88) & (contrast_m < 0.04),
            }

            for name, module in self.alpha_modules:
                alpha = getattr(module, "_last_alpha", None)
                if alpha is None:
                    continue
                a = alpha.detach().float()
                if a.shape[1] != 1:
                    a = a.mean(dim=1, keepdim=True)
                stage = STAGE_MAP.get(name, name)
                append_vals(self.values, stage, "all_pixels", a)
                self.pixel_counts.setdefault(stage, {})["all_pixels"] = self.pixel_counts.setdefault(stage, {}).get("all_pixels", 0) + int(a.numel())
                for cat, mask_m in masks_m.items():
                    mask_a = F.interpolate(mask_m.float(), size=a.shape[-2:], mode="nearest") > 0.5
                    self.pixel_counts.setdefault(stage, {})[cat] = self.pixel_counts.setdefault(stage, {}).get(cat, 0) + int(mask_a.sum().item())
                    append_vals(self.values, stage, cat, a, mask_a)
            super().update_metrics(preds, batch)

        def finalize_metrics(self):
            rows = []
            for stage in ["P3/8", "P4/16", "P5/32"]:
                for cat in CATEGORIES:
                    chunks = self.values.get(stage, {}).get(cat, [])
                    count = self.pixel_counts.get(stage, {}).get(cat, 0)
                    if not chunks:
                        rows.append({
                            "model": model_name,
                            "stage": stage,
                            "category": cat,
                            "n_alpha_pixels": count,
                            "alpha_mean": "",
                            "alpha_p25": "",
                            "alpha_p50": "",
                            "alpha_p75": "",
                            "alpha_p95": "",
                            "depth_weight_mean": "",
                        })
                        continue
                    vals = torch.cat(chunks)
                    q = torch.quantile(vals, torch.tensor([0.25, 0.5, 0.75, 0.95]))
                    mean = float(vals.mean())
                    rows.append({
                        "model": model_name,
                        "stage": stage,
                        "category": cat,
                        "n_alpha_pixels": int(vals.numel()),
                        "alpha_mean": mean,
                        "alpha_p25": float(q[0]),
                        "alpha_p50": float(q[1]),
                        "alpha_p75": float(q[2]),
                        "alpha_p95": float(q[3]),
                        "depth_weight_mean": 1.0 - mean,
                    })
            alpha_rows_holder.extend(rows)
            print("ALPHA_ROWS", model_name, json.dumps(rows, ensure_ascii=False, indent=2))
            super().finalize_metrics()

    return AlphaBlendValidator


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    alpha_rows = []
    blend_rows = []
    for run in RUNS:
        print("\n" + "=" * 100)
        print("RUN", run["model"])
        print("WEIGHTS", run["weights"])
        model = YOLO(run["weights"])
        validator = make_validator(run["model"], blend_rows, alpha_rows)
        model.val(
            validator=validator,
            data=DATA,
            imgsz=640,
            batch=4,
            device=0,
            workers=2,
            conf=0.55,
            iou=0.7,
            save_txt=False,
            save_json=False,
            plots=False,
            verbose=False,
            project="/workspace/genye_rgbd_route_probe/alpha_stats/valtmp",
            name=f"{run['model']}_on_val_expaug_alpha_blend",
            exist_ok=True,
        )
        write_csv(OUT_ALPHA_CSV, alpha_rows)
        write_csv(OUT_BLEND_CSV, blend_rows)
        OUT_JSON.write_text(json.dumps({"alpha": alpha_rows, "blend": blend_rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nOUT_ALPHA_CSV", OUT_ALPHA_CSV)
    print("OUT_BLEND_CSV", OUT_BLEND_CSV)
    print("OUT_JSON", OUT_JSON)
    print("BLEND_ROWS", json.dumps(blend_rows, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
