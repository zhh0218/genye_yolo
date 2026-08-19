import sys
from ultralytics import YOLO
w, name = sys.argv[1], sys.argv[2]
m = YOLO(w)
metrics = m.val(data="/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml",
    split="val", imgsz=640, batch=16, device=3, workers=4, verbose=False,
    project="/workspace/ultralytics-main_for_genye/runs/segment", name=name, exist_ok=True)
print("OVERALL_MASK", round(float(metrics.seg.map),5), round(float(metrics.seg.map50),5))
print("OVERALL_BOX", round(float(metrics.box.map),5), round(float(metrics.box.map50),5))
