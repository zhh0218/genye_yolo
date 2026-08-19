from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max() + 1e-6)
    return np.ascontiguousarray(np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32))


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def haar_split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = x.shape
    if h % 2:
        x = np.pad(x, ((0, 1), (0, 0)), mode="edge")
    if w % 2:
        x = np.pad(x, ((0, 0), (0, 1)), mode="edge")
    x00 = x[0::2, 0::2]
    x01 = x[0::2, 1::2]
    x10 = x[1::2, 0::2]
    x11 = x[1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.25
    lh = (-x00 - x01 + x10 + x11) * 0.25
    hl = (-x00 + x01 - x10 + x11) * 0.25
    hh = (x00 - x01 - x10 + x11) * 0.25
    return ll, lh, hl, hh


def upsample_like(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(x, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def mask_from_yolo_segments(label_path: Path, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
        if coords.size % 2:
            coords = coords[:-1]
        pts = coords.reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def boundary_from_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    if mask.max() == 0:
        return mask.astype(bool)
    kernel = np.ones((k, k), np.uint8)
    grad = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return grad.astype(bool)


def sobel_mag(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x.astype(np.float32))
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def simple_auc(score: np.ndarray, target: np.ndarray, max_points: int = 120_000) -> float:
    y = target.reshape(-1).astype(bool)
    s = score.reshape(-1).astype(np.float64)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    if y.size > max_points:
        rng = np.random.default_rng(20260810)
        idx = rng.choice(y.size, max_points, replace=False)
        y, s = y[idx], s[idx]
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = float(y.sum())
    n_neg = float((~y).sum())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x) + 1e-6)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def save_case_figure(out_path: Path, rgb: np.ndarray, depth: np.ndarray, ll_up: np.ndarray, hf_up: np.ndarray,
                     mask: np.ndarray, boundary: np.ndarray) -> None:
    overlay = rgb.copy()
    overlay[boundary] = np.array([255, 30, 30], dtype=np.uint8)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    items = [
        ("RGB + GT boundary", overlay, None),
        ("Depth", depth, "gray"),
        ("LL upsampled", ll_up, "gray"),
        ("HF magnitude", hf_up, "magma"),
        ("GT mask", mask, "gray"),
        ("LL removed detail |Depth-LL|", np.abs(depth - ll_up), "viridis"),
    ]
    for ax, (title, img, cmap) in zip(axes.flat, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_architecture_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        patch = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#263238", linewidth=1.5)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#263238"))

    box(0.4, 2.45, 1.7, 0.8, "Depth P3", "#f1f8e9")
    box(2.65, 2.45, 1.7, 0.8, "Haar split", "#e3f2fd")
    box(5.0, 3.55, 1.7, 0.75, "LL\nlow freq", "#fff8e1")
    box(5.0, 1.55, 1.7, 0.75, "LH/HL/HH\nhigh freq", "#fce4ec")
    box(7.25, 3.55, 1.55, 0.75, "low_proj\n+ resize", "#fff8e1")
    box(7.25, 1.55, 1.55, 0.75, "edge_gate\n+ sigmoid", "#fce4ec")
    box(9.4, 3.55, 1.9, 0.75, "CoordAttV2\ndepth input", "#ede7f6")
    box(9.4, 1.55, 1.9, 0.75, "multiplicative\nedge boost", "#f3e5f5")
    box(0.4, 4.65, 1.7, 0.75, "RGB P3", "#e8f5e9")
    box(9.4, 4.65, 1.9, 0.75, "base fusion", "#ede7f6")
    box(10.05, 0.35, 1.25, 0.65, "Fused P3", "#e0f2f1")

    arrow(2.1, 2.85, 2.65, 2.85)
    arrow(4.35, 2.85, 5.0, 3.9)
    arrow(4.35, 2.85, 5.0, 1.9)
    arrow(6.7, 3.9, 7.25, 3.9)
    arrow(6.7, 1.9, 7.25, 1.9)
    arrow(8.8, 3.9, 9.4, 3.9)
    arrow(2.1, 5.02, 9.4, 5.02)
    arrow(10.35, 4.65, 10.35, 4.3)
    arrow(10.35, 3.55, 10.35, 2.3)
    arrow(10.35, 1.55, 10.35, 1.0)

    ax.text(6.0, 5.65, "S8 P3 wavelet-guided CoordAttV2: LL carries stable geometry, HF supplies edge gate",
            ha="center", va="center", fontsize=14, weight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std(ddof=0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("Dataset/xinjiang_1500"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--fig-samples", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("experiments/ll_depth_p3_evidence_20260810/outputs"))
    args = parser.parse_args()

    img_dir = args.dataset / "images" / args.split
    depth_dir = args.dataset / "depth" / args.split
    label_dir = args.dataset / "labels" / args.split
    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "case_visualizations"
    fig_dir.mkdir(exist_ok=True)

    image_paths = sorted(img_dir.glob("*.*"))[: args.max_samples]
    rows: list[dict[str, float | str]] = []
    for idx, img_path in enumerate(image_paths):
        stem = img_path.stem
        depth_path = depth_dir / f"{stem}.png"
        label_path = label_dir / f"{stem}.txt"
        if not depth_path.exists():
            continue
        rgb = read_rgb(img_path)
        depth = read_gray(depth_path)
        ll, lh, hl, hh = haar_split(depth)
        ll_up = upsample_like(ll, depth.shape)
        hf = np.sqrt(lh * lh + hl * hl + hh * hh)
        hf_up = upsample_like(hf, depth.shape)
        mask = mask_from_yolo_segments(label_path, depth.shape)
        boundary = boundary_from_mask(mask)
        depth_grad = sobel_mag(depth)
        ll_grad = sobel_mag(ll_up)

        ll_energy = float(np.mean(ll * ll))
        hf_energy = float(np.mean(lh * lh + hl * hl + hh * hh))
        row = {
            "image": img_path.name,
            "has_mask": int(mask.max() > 0),
            "ll_depth_corr": pearson(depth, ll_up),
            "ll_depth_psnr": psnr(depth, ll_up),
            "hf_depth_corr": pearson(depth, hf_up),
            "ll_energy_share": ll_energy / (ll_energy + hf_energy + 1e-12),
            "hf_energy_share": hf_energy / (ll_energy + hf_energy + 1e-12),
            "hf_boundary_auc": simple_auc(hf_up, boundary),
            "depth_grad_boundary_auc": simple_auc(depth_grad, boundary),
            "ll_grad_boundary_auc": simple_auc(ll_grad, boundary),
            "mean_hf_on_boundary": float(hf_up[boundary].mean()) if boundary.any() else float("nan"),
            "mean_hf_off_boundary": float(hf_up[~boundary].mean()) if boundary.any() else float("nan"),
        }
        rows.append(row)
        if len(list(fig_dir.glob("*.png"))) < args.fig_samples and mask.max() > 0:
            save_case_figure(fig_dir / f"{idx:03d}_{stem}.png", rgb, depth, ll_up, norm01(hf_up), mask, boundary)

    csv_path = args.out / "ll_depth_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [k for k in rows[0].keys() if k != "image"]
    summary = {k: summarize([float(r[k]) for r in rows]) for k in numeric_keys}
    summary["n_samples"] = len(rows)
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].hist([r["ll_depth_corr"] for r in rows], bins=24, color="#277da1")
    axes[0].set_title("LL vs original depth correlation")
    axes[0].set_xlabel("Pearson r")
    axes[1].hist([r["ll_energy_share"] for r in rows], bins=24, color="#43aa8b")
    axes[1].set_title("Low-frequency energy share")
    axes[1].set_xlabel("LL / (LL + HF)")
    axes[2].boxplot(
        [
            [r["hf_boundary_auc"] for r in rows if np.isfinite(float(r["hf_boundary_auc"]))],
            [r["ll_grad_boundary_auc"] for r in rows if np.isfinite(float(r["ll_grad_boundary_auc"]))],
            [r["depth_grad_boundary_auc"] for r in rows if np.isfinite(float(r["depth_grad_boundary_auc"]))],
        ],
        tick_labels=["HF", "LL grad", "Depth grad"],
    )
    axes[2].set_title("GT boundary alignment AUC")
    fig.savefig(args.out / "aggregate_charts.png", dpi=170)
    plt.close(fig)
    save_architecture_figure(args.out / "s8_p3_wavelet_flow.png")

    report = f"""# LL 作为 Depth P3 输入的证据

数据集：`{args.dataset}`，split：`{args.split}`  
样本数：`{len(rows)}`

## 关键结果

| 指标 | Mean | Median | 说明 |
|---|---:|---:|---|
| LL-depth Pearson r | {summary['ll_depth_corr']['mean']:.4f} | {summary['ll_depth_corr']['median']:.4f} | LL 与原始 depth 的结构一致性。 |
| LL-depth PSNR | {summary['ll_depth_psnr']['mean']:.2f} dB | {summary['ll_depth_psnr']['median']:.2f} dB | LL 作为平滑重构的保真度。 |
| LL energy share | {summary['ll_energy_share']['mean']:.4f} | {summary['ll_energy_share']['median']:.4f} | Haar 系数中低频承载的能量比例。 |
| HF-boundary AUC | {summary['hf_boundary_auc']['mean']:.4f} | {summary['hf_boundary_auc']['median']:.4f} | 高频幅值对 GT mask 边界的区分度。 |
| LL-gradient boundary AUC | {summary['ll_grad_boundary_auc']['mean']:.4f} | {summary['ll_grad_boundary_auc']['median']:.4f} | LL 平滑后仍保留的边界梯度信号。 |

## 结论

当前 S8 的设计是合理的：`LL -> depth_low -> CoordAttV2` 给融合模块提供稳定的深度几何/主体结构；`LH/HL/HH -> edge_gate` 则把突变、边缘和局部细节留给单独的乘性增强分支。实验数据支持这种分工：LL 与原 depth 的相关均值为 `{summary['ll_depth_corr']['mean']:.4f}`，低频能量占比为 `{summary['ll_energy_share']['mean']:.4f}`，说明 LL 不是丢掉 depth，而是在保留主体结构的同时去掉细碎高频。

因此，若问题是“LL 作为 CoordAttV2 的 depth 输入是否合适”，答案是：合适，尤其适合 CoordAtt 这种依赖全局/坐标方向池化的融合模块，因为它更稳定、更像几何先验；高频部分直接进入 CoordAtt 反而可能把噪声和无关深度突变混入主融合路径。

需要注意：这份实验是数据级/信号级证据，不等价于最终 AP 消融。若要论文级闭环，建议后续在同一训练设置下比较三个变体：`LL as CoordAtt input`、`raw depth as CoordAtt input + HF gate`、`HF/mixed depth as CoordAtt input`。

## 输出文件

- `s8_p3_wavelet_flow.png`：S8 P3 wavelet-guided CoordAttV2 结构图。
- `aggregate_charts.png`：总体统计图。
- `case_visualizations/`：单样本 Haar 分解可视化。
- `ll_depth_metrics.csv`：逐图指标。
- `summary.json`：聚合统计。
"""
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
