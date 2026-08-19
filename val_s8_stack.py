from ultralytics import YOLO
m = YOLO("/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt")
metrics = m.val(
    data="/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml",
    split="val", imgsz=640, batch=16, device=3, workers=4, verbose=False,
    project="/workspace/ultralytics-main_for_genye/runs/segment", name="s8_stack_eval", exist_ok=True,
)
try:
    print("OVERALL_MASK", "map5095", round(float(metrics.seg.map),5), "map50", round(float(metrics.seg.map50),5))
    print("OVERALL_BOX", "map5095", round(float(metrics.box.map),5), "map50", round(float(metrics.box.map50),5))
except Exception as e:
    print("ERR_METRIC", e)
