import sys

from ultralytics import YOLO

w, name = sys.argv[1], sys.argv[2]
m = YOLO(w)
r = m.val(
    data="/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml",
    split="val",
    imgsz=640,
    batch=16,
    device=3,
    workers=4,
    verbose=False,
    project="/workspace/ultralytics-main_for_genye/runs/segment",
    name=name,
    exist_ok=True,
)


def pc(metric, label):
    idx = list(metric.ap_class_index)
    if 1 in idx:
        i = idx.index(1)
        p, rc, ap50, ap = metric.class_result(i)
        print(f"RESULT {name} {label} CLS01 P={p:.4f} R={rc:.4f} AP50={ap50:.4f} AP5095={ap:.4f}")
    print(f"RESULT {name} {label} ALL  AP50={metric.map50:.4f} AP5095={metric.map:.4f}")


pc(r.box, "BOX")
pc(r.seg, "MASK")
