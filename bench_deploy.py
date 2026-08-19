import statistics
import time

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

G = "/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye"
T = "/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-genye-teacher-ablation"
MODELS = [
    ("早期FRM/FFM", T + "/YOLO11s-genye-ablation-old-cmx-ffm-stable/weights/best.pt"),
    ("CALF", G + "/coord_attv2-s7/weights/best.pt"),
    ("新方法CAF+WAFF+AuxFPN", G + "/coord_attv2-s8/weights/best.pt"),
    ("+逐区域门控", G + "/coord_attv2-gatedfull/weights/best.pt"),
]
for tag, w in MODELS:
    m = YOLO(w)
    params_m = sum(p.numel() for p in m.model.parameters()) / 1e6
    try:
        gflops = get_flops(m.model, imgsz=640)
    except Exception:
        gflops = -1
    ms = "NA"
    try:
        ch = int(getattr(m.model, "yaml", {}).get("ch", 3))
        net = m.model.eval().to("cuda:3")
        x = torch.randn(1, ch, 640, 640, device="cuda:3")
        with torch.no_grad():
            for _ in range(15):
                net(x)
            torch.cuda.synchronize(3)
            ts = []
            for _ in range(80):
                torch.cuda.synchronize(3)
                t0 = time.perf_counter()
                net(x)
                torch.cuda.synchronize(3)
                ts.append((time.perf_counter() - t0) * 1000)
        ms = f"{statistics.median(ts):.2f}"
        net.to("cpu")
        del net, x
        torch.cuda.empty_cache()
    except Exception as e:
        ms = f"ERR:{str(e)[:50]}"
    fps = ("%.1f" % (1000 / float(ms))) if ms.replace(".", "").isdigit() else "NA"
    print("DEPLOY %-22s params=%.2fM gflops=%.1f infer_ms=%s fps=%s" % (tag, params_m, gflops, ms, fps))
