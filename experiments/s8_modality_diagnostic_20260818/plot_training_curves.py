from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                key = key.strip()
                if not key:
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
        return rows


def _series(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if key in r]


def main() -> None:
    root = _repo_root()
    exp_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csv", default=str(root / "YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/results.csv")
    )
    parser.add_argument("--output-dir", default=str(exp_dir / "outputs"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(Path(args.results_csv))
    epochs = _series(rows, "epoch")

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=160)
    ax.plot(epochs, _series(rows, "metrics/mAP50-95(B)"), label="Box mAP50-95", color="#2f6f8f", linewidth=2.0)
    ax.plot(epochs, _series(rows, "metrics/mAP50-95(M)"), label="Mask mAP50-95", color="#b05a37", linewidth=2.0)
    ax.plot(epochs, _series(rows, "metrics/mAP50(B)"), label="Box mAP50", color="#6fa8dc", linewidth=1.3, alpha=0.85)
    ax.plot(epochs, _series(rows, "metrics/mAP50(M)"), label="Mask mAP50", color="#e6a06f", linewidth=1.3, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("S8 validation mAP during original training")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "s8_training_map_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=160)
    for key, label, color in [
        ("train/box_loss", "train box", "#2f6f8f"),
        ("train/seg_loss", "train seg", "#b05a37"),
        ("train/cls_loss", "train cls", "#5d8f47"),
        ("val/box_loss", "val box", "#6fa8dc"),
        ("val/seg_loss", "val seg", "#e6a06f"),
        ("val/cls_loss", "val cls", "#94bd79"),
    ]:
        ax.plot(epochs, _series(rows, key), label=label, linewidth=1.6, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("S8 training and validation losses")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "s8_training_loss_curves.png")
    plt.close(fig)

    best_box = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
    best_mask = max(rows, key=lambda r: r["metrics/mAP50-95(M)"])
    final = rows[-1]
    summary = {
        "results_csv": str(Path(args.results_csv).resolve()),
        "epochs": len(rows),
        "best_box_epoch": int(best_box["epoch"]),
        "best_box_map": round(best_box["metrics/mAP50-95(B)"], 6),
        "best_mask_epoch": int(best_mask["epoch"]),
        "best_mask_map": round(best_mask["metrics/mAP50-95(M)"], 6),
        "final_epoch": int(final["epoch"]),
        "final_box_map": round(final["metrics/mAP50-95(B)"], 6),
        "final_mask_map": round(final["metrics/mAP50-95(M)"], 6),
    }
    (out_dir / "training_curve_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
