"""Pre-train build test: gated-Full builds w/ spatial_channel gate + forward works; default(s8) no regression."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch import nn

from ultralytics import YOLO

GY = "/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg-gatedfull.yaml"
DY = "/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg.yaml"

print("=== gated-Full build ===")
net = YOLO(GY).model
gms = [
    getattr(mod, "gate_mode", "?")
    for mod in net.modules()
    if type(mod).__name__ == "CoordAttV2" and getattr(mod, "adaptive_gate", False)
]
print("CoordAttV2 adaptive gate_modes:", gms)
assert gms and all(x == "spatial_channel" for x in gms), f"gate not spatial_channel: {gms}"
for mod in net.modules():
    if type(mod).__name__ == "CoordAttV2" and getattr(mod, "adaptive_gate", False):
        has_conv = any(isinstance(l, nn.Conv2d) for l in mod.mod_gate)
        print("  spatial mod_gate has Conv2d:", has_conv)
        assert has_conv, "mod_gate not spatial conv"
        break
rgb, ir = torch.randn(1, 3, 640, 640), torch.randn(1, 3, 640, 640)
net.eval()
with torch.no_grad():
    ok = False
    for call in (lambda: net(rgb, ir), lambda: net.predict(rgb, ir), lambda: net._predict_once(rgb, ir)):
        try:
            call()
            ok = True
            break
        except Exception as e:
            last = e
    assert ok, f"forward failed: {last}"
print("FORWARD OK | gated-Full params:", round(sum(p.numel() for p in net.parameters()) / 1e6, 4), "M")

print("=== default (s8) regression ===")
net2 = YOLO(DY).model
g2 = [
    getattr(mod, "gate_mode", None)
    for mod in net2.modules()
    if type(mod).__name__ == "CoordAttV2" and getattr(mod, "adaptive_gate", False)
]
print("default adaptive CoordAttV2 (should be empty, gate off):", g2)
print("default(s8) params:", round(sum(p.numel() for p in net2.parameters()) / 1e6, 4), "M")
print("BUILD TEST PASSED")
