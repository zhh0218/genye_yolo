from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/ultralytics-main_for_genye_release")
from ultralytics import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator

WEIGHTS = Path("/workspace/ultralytics-main_for_genye_release/YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/xinjiang_1500_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt")
DATA = Path("/workspace/ultralytics-main_for_genye_release/Dataset/xinjiang_1500/data_3cls.yaml")
OUT_DIR = Path("/workspace/genye_rgbd_route_probe/alpha_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "clean_xinjiang1500_best_val300_spatial_illum_alpha_stats.json"
OUT_CSV = OUT_DIR / "clean_xinjiang1500_best_val300_spatial_illum_alpha_stats.csv"

STAGE_MAP = {
    "model.36.coord_att": "P3/8",
    "model.37.coord_att": "P4/16",
    "model.38.coord_att": "P5/32",
}

class AlphaStatsValidator(SegmentationValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.alpha_modules = []
        for name, module in model.named_modules():
            if type(module).__name__ == "CoordAttV2" and getattr(module, "adaptive_gate", False):
                self.alpha_modules.append((name, module))
        self.alpha_values = {name: [] for name, _ in self.alpha_modules}
        self.alpha_batch_means = {name: [] for name, _ in self.alpha_modules}
        self.blend = {}
        for name, module in self.alpha_modules:
            if hasattr(module, "blend_logits"):
                self.blend[name] = torch.softmax(module.blend_logits.detach().cpu(), dim=0).tolist()

    def update_metrics(self, preds, batch):
        for name, module in self.alpha_modules:
            alpha = getattr(module, "_last_alpha", None)
            if alpha is None:
                continue
            a = alpha.detach().float().cpu()
            self.alpha_values[name].append(a.reshape(-1))
            # per-image mean, useful to know image-to-image variation
            self.alpha_batch_means[name].append(a.flatten(1).mean(dim=1))
        super().update_metrics(preds, batch)

    def finalize_metrics(self):
        summary = []
        for name, _module in self.alpha_modules:
            vals = torch.cat(self.alpha_values[name]) if self.alpha_values[name] else torch.empty(0)
            img_means = torch.cat(self.alpha_batch_means[name]) if self.alpha_batch_means[name] else torch.empty(0)
            if vals.numel() == 0:
                continue
            q = torch.quantile(vals, torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
            row = {
                "module": name,
                "stage": STAGE_MAP.get(name, name),
                "n_values": int(vals.numel()),
                "n_images": int(img_means.numel()),
                "alpha_mean": float(vals.mean()),
                "alpha_std": float(vals.std(unbiased=False)),
                "alpha_min": float(vals.min()),
                "alpha_p01": float(q[0]),
                "alpha_p05": float(q[1]),
                "alpha_p25": float(q[2]),
                "alpha_p50": float(q[3]),
                "alpha_p75": float(q[4]),
                "alpha_p95": float(q[5]),
                "alpha_p99": float(q[6]),
                "alpha_max": float(vals.max()),
                "per_image_mean_mean": float(img_means.mean()),
                "per_image_mean_std": float(img_means.std(unbiased=False)),
                "blend_base": float(self.blend.get(name, [float('nan'), float('nan')])[0]),
                "blend_out": float(self.blend.get(name, [float('nan'), float('nan')])[1]),
            }
            summary.append(row)

        with OUT_JSON.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        print("ALPHA_STATS_JSON", OUT_JSON)
        print("ALPHA_STATS_CSV", OUT_CSV)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        super().finalize_metrics()


def main():
    model = YOLO(str(WEIGHTS))
    model.val(
        validator=AlphaStatsValidator,
        data=str(DATA),
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        conf=0.55,
        iou=0.7,
        save_txt=False,
        save_json=False,
        plots=False,
        verbose=False,
        project="/workspace/genye_rgbd_route_probe/alpha_stats/valtmp",
        name="clean_xinjiang1500_best_val300_alpha",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
