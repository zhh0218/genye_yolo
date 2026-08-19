import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

DATA = ROOT / "examples" / "quick_start" / "quick_start_rgbd.yaml"
WEIGHTS = ROOT / "YOLOv11-RGB-D-coord_attv2-genye" / "coord_attv2-s8" / "weights" / "best.pt"


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing data config: {DATA}")
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing final weights: {WEIGHTS}")

    model = YOLO(str(WEIGHTS))
    model.val(
        data=str(DATA),
        imgsz=640,
        batch=8,
        device="0",
        split="val",
        project=str(ROOT / "examples" / "quick_start" / "runs"),
        name="smoke_val",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
