from __future__ import annotations

import argparse
import csv
import json
import types
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO


MODES = {
    "RD": (1.0, 1.0),
    "R-only": (1.0, 0.0),
    "D-only": (0.0, 1.0),
    "Empty": (0.0, 0.0),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_float(value):
    try:
        return round(float(value), 6)
    except Exception:
        return None


def _install_feature_mute(yolo: YOLO) -> list[int]:
    """Patch only this in-memory model so fusion receives lambda-scaled RGB/depth features."""
    base_model = yolo.model
    _backfill_old_checkpoint_attrs(base_model)
    if not hasattr(base_model, "_fusion_indices"):
        ds, de = base_model._depth_range
        fusion_start = de + (1 if bool(getattr(base_model, "depth_fpn", False)) else 0)
        base_model._fusion_indices = [fusion_start + i for i in range(3)]

    for idx in base_model._fusion_indices:
        module = base_model.model[idx]
        if not hasattr(module, "_modality_original_forward"):
            module._modality_original_forward = module.forward
            module._modality_rgb_lambda = 1.0
            module._modality_depth_lambda = 1.0

            def scaled_forward(self, rgb, depth):
                rgb_lambda = float(getattr(self, "_modality_rgb_lambda", 1.0))
                depth_lambda = float(getattr(self, "_modality_depth_lambda", 1.0))
                return self._modality_original_forward(rgb * rgb_lambda, depth * depth_lambda)

            module.forward = types.MethodType(scaled_forward, module)
    return list(base_model._fusion_indices)


def _backfill_old_checkpoint_attrs(base_model) -> None:
    """Keep old S8 checkpoints compatible with newer fusion code defaults."""
    for module in base_model.modules():
        if module.__class__.__name__ == "CoordAttV2":
            if not hasattr(module, "adaptive_gate"):
                module.adaptive_gate = False
            if not hasattr(module, "gate_mode"):
                module.gate_mode = "channel"
            if not hasattr(module, "base_scale"):
                module.base_scale = 1.0
            if not hasattr(module, "out_scale"):
                module.out_scale = 1.0
            if not hasattr(module, "learnable_blend"):
                module.learnable_blend = False


def _set_mode(yolo: YOLO, rgb_lambda: float, depth_lambda: float) -> None:
    for idx in yolo.model._fusion_indices:
        module = yolo.model.model[idx]
        module._modality_rgb_lambda = float(rgb_lambda)
        module._modality_depth_lambda = float(depth_lambda)


def _extract_metrics(metrics) -> dict[str, float | None]:
    return {
        "box_precision": _as_float(getattr(metrics.box, "mp", None)),
        "box_recall": _as_float(getattr(metrics.box, "mr", None)),
        "box_map50": _as_float(getattr(metrics.box, "map50", None)),
        "box_map": _as_float(getattr(metrics.box, "map", None)),
        "mask_precision": _as_float(getattr(metrics.seg, "mp", None)),
        "mask_recall": _as_float(getattr(metrics.seg, "mr", None)),
        "mask_map50": _as_float(getattr(metrics.seg, "map50", None)),
        "mask_map": _as_float(getattr(metrics.seg, "map", None)),
    }


def _shapley(results: dict[str, dict[str, float | None]], metric: str) -> dict[str, float | None]:
    v_rd = results["RD"].get(metric)
    v_r = results["R-only"].get(metric)
    v_d = results["D-only"].get(metric)
    v_0 = results["Empty"].get(metric)
    if None in (v_rd, v_r, v_d, v_0):
        return {"metric": metric, "rgb_contribution": None, "depth_contribution": None, "interaction": None}

    c_r = 0.5 * ((v_r - v_0) + (v_rd - v_d))
    c_d = 0.5 * ((v_d - v_0) + (v_rd - v_r))
    gain = v_rd - v_0
    interaction = v_rd - v_r - v_d + v_0
    return {
        "metric": metric,
        "rgb_contribution": round(c_r, 6),
        "depth_contribution": round(c_d, 6),
        "rgb_ratio": round(c_r / gain, 6) if abs(gain) > 1e-12 else None,
        "depth_ratio": round(c_d / gain, 6) if abs(gain) > 1e-12 else None,
        "interaction": round(interaction, 6),
        "efficiency_check": round((c_r + c_d) - gain, 9),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_mode_bars(results: dict[str, dict], out_dir: Path) -> None:
    modes = list(MODES)
    x = range(len(modes))
    box = [results[m]["box_map"] for m in modes]
    mask = [results[m]["mask_map"] for m in modes]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    ax.bar([i - 0.18 for i in x], box, width=0.36, label="Box mAP50-95", color="#2f6f8f")
    ax.bar([i + 0.18 for i in x], mask, width=0.36, label="Mask mAP50-95", color="#b05a37")
    ax.set_xticks(list(x), modes)
    ax.set_ylim(0, max(0.05, max(box + mask) * 1.18))
    ax.set_ylabel("mAP50-95")
    ax.set_title("S8 four-state validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "four_state_map_bars.png")
    plt.close(fig)


def _plot_shapley(shapley_rows: list[dict], out_dir: Path) -> None:
    rows = [r for r in shapley_rows if r["metric"] in {"box_map", "mask_map"}]
    labels = ["Box mAP50-95" if r["metric"] == "box_map" else "Mask mAP50-95" for r in rows]
    rgb = [r["rgb_contribution"] for r in rows]
    depth = [r["depth_contribution"] for r in rows]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
    ax.bar(x, rgb, label="RGB Shapley", color="#2f6f8f")
    ax.bar(x, depth, bottom=rgb, label="Depth Shapley", color="#b05a37")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Contribution")
    ax.set_title("RGB / Depth Shapley contribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "shapley_contribution_stacked.png")
    plt.close(fig)


def main() -> None:
    root = _repo_root()
    exp_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(root / "YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt"))
    parser.add_argument("--data", default=str(exp_dir / "data_3cls_local.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--output-dir", default=str(exp_dir / "outputs"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.weights)
    fusion_indices = _install_feature_mute(yolo)

    results: dict[str, dict] = {}
    mode_rows = []
    for mode, (rgb_lambda, depth_lambda) in MODES.items():
        _set_mode(yolo, rgb_lambda, depth_lambda)
        metrics = yolo.val(
            data=args.data,
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            verbose=False,
            plots=args.plots,
            conf=args.conf,
            iou=args.iou,
            project=str(out_dir / "val_runs"),
            name=mode.replace("-", "_"),
            exist_ok=True,
        )
        row = {
            "mode": mode,
            "rgb_lambda": rgb_lambda,
            "depth_lambda": depth_lambda,
            **_extract_metrics(metrics),
        }
        results[mode] = row
        mode_rows.append(row)

    metric_names = ["box_map", "box_map50", "mask_map", "mask_map50"]
    shapley_rows = [_shapley(results, metric) for metric in metric_names]

    payload = {
        "experiment": "s8_modality_diagnostic_20260818",
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "fusion_indices": fusion_indices,
        "modes": results,
        "shapley": shapley_rows,
    }
    (out_dir / "four_state_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "four_state_metrics.csv", mode_rows)
    _write_csv(out_dir / "shapley_metrics.csv", shapley_rows)
    _plot_mode_bars(results, out_dir)
    _plot_shapley(shapley_rows, out_dir)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
