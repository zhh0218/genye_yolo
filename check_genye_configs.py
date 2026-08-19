"""Verify coord_attv2-s3/s7/s8 are DIFFERENT module configs (not seeds): compare params + module types."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import Counter

from ultralytics import YOLO

cnt = lambda m: sum(p.numel() for p in m.parameters())
R = "/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye"
for s in ["s3", "s7", "s8"]:
    p = f"{R}/coord_attv2-{s}/weights/best.pt"
    net = YOLO(p).model
    tot = cnt(net)
    ty = Counter(type(m).__name__ for m in net.modules())
    wav = sum(v for k, v in ty.items() if "Wavelet" in k or "WAFF" in k)
    fpn = sum(v for k, v in ty.items() if "FPN" in k or "DepthLight" in k)
    coord = sum(v for k, v in ty.items() if "CoordAtt" in k)
    gate = sum(v for k, v in ty.items() if "Gate" in k or "Adaptive" in k)
    fus = [k for k in ty if any(x in k for x in ("Wavelet", "CoordAtt", "FPN", "Fusion", "Gate", "Concat"))]
    print(f"{s}: params {tot / 1e6:.4f}M | wavelet={wav} fpn={fpn} coordatt={coord} gate/adaptive={gate}")
    print(f"     fusion-related module types: {sorted(fus)}")
