import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

DATA = ROOT / "examples" / "quick_start" / "quick_start_rgbd.yaml"
MODEL_YAML = ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11-seg.yaml"
PRETRAINED = ROOT / "yolo11l.pt"


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing data config: {DATA}")
    if not MODEL_YAML.exists():
        raise FileNotFoundError(f"Missing model config: {MODEL_YAML}")
    if not PRETRAINED.exists():
        raise FileNotFoundError(f"Missing pretrained weights: {PRETRAINED}")

    model = YOLO(str(MODEL_YAML))
    model.load(str(PRETRAINED))
    model.train(
        data=str(DATA),
        imgsz=640,
        epochs=2,
        batch=4,
        workers=2,
        device="0",
        optimizer="MuSGD",
        amp=True,
        project=str(ROOT / "examples" / "quick_start" / "runs"),
        name="smoke_train",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
