# -*- coding: utf-8 -*-
"""Port spatial_channel region-gate (gate_mode) from ultralytics-main into ultralytics-main_for_genye.
   Additive: gate_mode defaults to 'channel' -> existing s3/s7/s8 behavior unchanged.
   Then create gated-Full model yaml + train script (copies, originals untouched)."""
import shutil, py_compile, os
FG = "/workspace/ultralytics-main_for_genye/ultralytics/nn/tasks.py"
MAIN = "/workspace/ultralytics-main/ultralytics/nn/tasks.py"
BAK = FG + ".bak_gateport"
if not os.path.exists(BAK):
    shutil.copy(FG, BAK); print("backup ->", BAK)
g = open(FG, encoding="utf-8").read()
m = open(MAIN, encoding="utf-8").read()

# E1: replace for_genye CoordAttV2 class body with main's (has gate_mode + spatial branch + _illum_prior)
mi = m.index("class CoordAttV2(nn.Module):"); mj = m.index("class RGBDCoordAttV2(nn.Module):")
main_cls = m[mi:mj]
assert 'gate_mode in ("spatial", "spatial_channel"' in main_cls, "main CoordAttV2 missing spatial branch"
gi = g.index("class CoordAttV2(nn.Module):"); gj = g.index("class RGBDCoordAttV2(nn.Module):")
g = g[:gi] + main_cls + g[gj:]

# E2: wrapper __init__ signatures (RGBDCoordAttV2 + RGBDWaveletGuidedCoordAttV2) add gate_mode
a = "def __init__(self, c_in, num_heads=8, kv_pool=10, adaptive_gate=False):"
b = 'def __init__(self, c_in, num_heads=8, kv_pool=10, adaptive_gate=False, gate_mode="channel"):'
assert g.count(a) == 2, f"E2 expected 2 got {g.count(a)}"; g = g.replace(a, b)

# E3: wrapper CoordAttV2 construction pass gate_mode
a2 = "self.coord_att = CoordAttV2(c_in * 2, c_in * 2, adaptive_gate=adaptive_gate)"
b2 = "self.coord_att = CoordAttV2(c_in * 2, c_in * 2, adaptive_gate=adaptive_gate, gate_mode=gate_mode)"
assert g.count(a2) == 2, f"E3 expected 2 got {g.count(a2)}"; g = g.replace(a2, b2)

# E4: build_model _fusion_kw add gate_mode from yaml
a3 = "_fusion_kw = dict(adaptive_gate=_adaptive_gate) if _adaptive_gate else {}"
b3 = "_fusion_kw = dict(adaptive_gate=_adaptive_gate, gate_mode=d.get('gate_mode', 'channel')) if _adaptive_gate else {}"
assert g.count(a3) == 1, f"E4 expected 1 got {g.count(a3)}"; g = g.replace(a3, b3)

# ensure F imported (main's _illum_prior uses F.avg_pool2d)
if "import torch.nn.functional as F" not in g and "from torch.nn import functional as F" not in g:
    g = g.replace("import torch\n", "import torch\nimport torch.nn.functional as F\n", 1)
    print("added F import")

open(FG, "w", encoding="utf-8").write(g)
py_compile.compile(FG, doraise=True)
print("PORT OK: tasks.py patched + compiles")

# gated-Full model yaml (copy + enable gate)
SRCY = "/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg.yaml"
DSTY = "/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg-gatedfull.yaml"
y = open(SRCY, encoding="utf-8").read()
assert y.count("modality_adaptive_gate: False") == 1
y = y.replace("modality_adaptive_gate: False # learned modality-priority gate: α·main + (1−α)·aux + attn_out",
              "modality_adaptive_gate: True # GATED-FULL: enable region gate\ngate_mode: 'spatial_channel' # spatial_channel region gate (ported from ultralytics-main)")
assert "gate_mode: 'spatial_channel'" in y and "modality_adaptive_gate: True" in y
open(DSTY, "w", encoding="utf-8").write(y); print("gated yaml ->", DSTY)

# gated train script (copy + point to gated yaml + new name)
SRCT = "/workspace/ultralytics-main_for_genye/train_genye.py"
DSTT = "/workspace/ultralytics-main_for_genye/train_genye_gatedfull.py"
t = open(SRCT, encoding="utf-8").read()
t = t.replace('model_yaml_path = r"/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg.yaml"',
              'model_yaml_path = r"/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg-gatedfull.yaml"')
t = t.replace("name='coord_attv2-s'", "name='coord_attv2-gatedfull'")
assert "yolo11-seg-gatedfull.yaml" in t and "coord_attv2-gatedfull" in t
open(DSTT, "w", encoding="utf-8").write(t); print("gated train script ->", DSTT)
print("ALL DONE")
