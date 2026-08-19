# spatial gate inference/validation experiment

This folder is isolated from the original project files. It contains a runnable script for validating the RGB-D dataset and exporting the spatial gate weights:

- `src_weight_*`: alpha in `alpha * src + (1 - alpha) * depth`
- `depth_weight_*`: `1 - alpha`
- `base_blend_weight`: learned weight for the gated base branch
- `out_blend_weight`: learned weight for the coordinate-attention output branch

Default dataset:

```powershell
D:\Graduate\project\genye\Dataset\xinjiang_1500_baoguang
```

Quick gate-weight check on the first 50 validation images:

```powershell
conda activate genye-yolo
cd D:\Graduate\project\genye
python experiments\spatial_gate_infer_validate_20260725-114839\infer_validate_spatial_gate.py --max-images 50
```

Run validation metrics too:

```powershell
python experiments\spatial_gate_infer_validate_20260725-114839\infer_validate_spatial_gate.py --run-val --max-images 50
```

Use another checkpoint, for example a gated/spatial checkpoint:

```powershell
python experiments\spatial_gate_infer_validate_20260725-114839\infer_validate_spatial_gate.py `
  --weights "D:\path\to\best.pt" `
  --max-images 50
```

Current defaults use:

```text
configs/yolo11-seg-spatialillum-learnblend-auxoff-xinjiang1500.yaml
YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/xinjiang_1500_baoguang_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt
```

Outputs are written under:

```text
experiments\spatial_gate_infer_validate_20260725-114839\outputs
```

The main file to inspect is:

```text
spatial_gate_rgb_depth_weights.csv
```
