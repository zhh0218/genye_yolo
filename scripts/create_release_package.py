from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

DEFAULT_ROOT = Path("/workspace/ultralytics-main_for_genye")
DEFAULT_RELEASE_NAME = "release_rgbd_s8"

IGNORE_NAMES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "wandb",
    "runs",
    "runtime_cfg",
}
IGNORE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".cache",
}


ROOT_FILES = [
    "README_RGBD_S8.md",
    "data-ubuntu-genye.yaml",
    "train_genye.py",
    "yolo11l.pt",
]

DOC_FILES = [
    "code_guide.md",
    "experiment_summary.md",
    "handover_checklist.md",
    "release_packaging.md",
]

SCRIPT_FILES = [
    "create_quick_start_subset.py",
    "create_release_package.py",
]

S8_FILES = [
    "args.yaml",
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxF1_curve.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "MaskF1_curve.png",
    "MaskPR_curve.png",
    "MaskP_curve.png",
    "MaskR_curve.png",
    "labels.jpg",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
    "val_batch1_labels.jpg",
    "val_batch1_pred.jpg",
    "val_batch2_labels.jpg",
    "val_batch2_pred.jpg",
]


def ignore_filter(_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        path = Path(name)
        if name in IGNORE_NAMES or path.suffix in IGNORE_SUFFIXES:
            ignored.add(name)
    return ignored


def ensure_inside_root(root: Path, path: Path) -> None:
    root = root.resolve()
    path = path.resolve()
    if root != path and root not in path.parents:
        raise RuntimeError(f"Unsafe path outside root: {path}")


def copy_file(root: Path, release: Path, rel_path: str, dst_rel_path: str | None = None, required: bool = True) -> None:
    src = root / rel_path
    dst = release / (dst_rel_path or rel_path)
    if not src.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {src}")
        print(f"Skip missing optional file: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(root: Path, release: Path, rel_path: str, required: bool = True) -> None:
    src = root / rel_path
    dst = release / rel_path
    if not src.exists():
        if required:
            raise FileNotFoundError(f"Missing required directory: {src}")
        print(f"Skip missing optional directory: {src}")
        return
    shutil.copytree(src, dst, ignore=ignore_filter)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(release: Path) -> None:
    rows = []
    for path in sorted(p for p in release.rglob("*") if p.is_file()):
        if path.name == "release_manifest.csv":
            continue
        rows.append(
            {
                "path": path.relative_to(release).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = release / "release_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote manifest: {manifest}")


def build_release(root: Path, release_name: str) -> Path:
    root = root.resolve()
    release = root / release_name
    ensure_inside_root(root, release)
    if release.exists():
        raise FileExistsError(
            f"Release directory already exists: {release}\nMove or remove it before creating a new package."
        )
    release.mkdir(parents=True)

    for rel in ROOT_FILES:
        copy_file(root, release, rel)

    copy_tree(root, release, "ultralytics")
    copy_tree(root, release, "examples/quick_start")

    for doc in DOC_FILES:
        copy_file(root, release, f"docs/{doc}")

    for script in SCRIPT_FILES:
        copy_file(root, release, f"scripts/{script}", required=False)

    log_rel = "LOG/genye11s_just_coord_att_v2+p3+dfpn.log"
    copy_file(root, release, log_rel)

    s8_rel = "YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8"
    copy_file(root, release, f"{s8_rel}/weights/best.pt")
    for rel in S8_FILES:
        copy_file(root, release, f"{s8_rel}/{rel}", required=False)

    write_manifest(release)
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean RGB-D S8 handover release package.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Project root directory.")
    parser.add_argument("--name", default=DEFAULT_RELEASE_NAME, help="Release directory name under root.")
    args = parser.parse_args()

    release = build_release(args.root, args.name)
    print(f"Created release package: {release}")
    print("Next check:")
    print(f"  cd {release}")
    print("  python examples/quick_start/quick_val.py")


if __name__ == "__main__":
    main()
