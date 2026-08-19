from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/workspace/ultralytics-main_for_genye_release')
from ultralytics import YOLO

DATA = '/workspace/ultralytics-main_for_genye_release/Dataset/xinjiang_1500_baoguang/data_3cls_val_expaug.yaml'
PROJECT = '/workspace/ultralytics-main_for_genye_release/YOLOv11-RGB-D-coord_attv2-genye/val_expaug_eval'
OUT_JSON = Path('/workspace/genye_rgbd_route_probe/eval_results/xinjiang1500_baoguang_val_expaug_3models_20260704.json')
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        'name': 'clean_xinjiang1500_best_on_xinjiang1500_baoguang_val_expaug',
        'weights': '/workspace/ultralytics-main_for_genye_release/YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/xinjiang_1500_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt',
        'legacy': False,
    },
    {
        'name': 'baoguang_xinjiang1500_baoguang_best_on_val_expaug',
        'weights': '/workspace/ultralytics-main_for_genye_release/YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/xinjiang_1500_baoguang_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt',
        'legacy': False,
    },
    {
        'name': 'shidi_s8_original_622_occlusion_best_on_xinjiang1500_baoguang_val_expaug',
        'weights': '/workspace/genye_rgbd_route_probe/weights/shidi_s8_original_622_occlusion_3class/best.pt',
        'legacy': True,
    },
]


def patch_legacy_coordattv2(model):
    patched = 0
    for module in model.modules():
        if type(module).__name__ != 'CoordAttV2':
            continue
        if not hasattr(module, 'adaptive_gate'):
            module.adaptive_gate = False
        if not hasattr(module, 'gate_mode'):
            module.gate_mode = 'channel'
        if not hasattr(module, 'base_scale'):
            module.base_scale = 1.0
        if not hasattr(module, 'out_scale'):
            module.out_scale = 1.0
        if not hasattr(module, 'learnable_blend'):
            module.learnable_blend = False
        patched += 1
    return patched


def to_float(x):
    if x is None:
        return None
    if hasattr(x, 'item'):
        return float(x.item())
    try:
        return float(x)
    except Exception:
        return None


def metric_summary(metrics):
    out = {}
    rd = getattr(metrics, 'results_dict', {}) or {}
    out['results_dict'] = {str(k): to_float(v) for k, v in rd.items()}
    for kind in ['box', 'seg']:
        m = getattr(metrics, kind, None)
        if m is None:
            continue
        kd = {}
        for attr in ['mp', 'mr', 'map50', 'map', 'map75']:
            if hasattr(m, attr):
                kd[attr] = to_float(getattr(m, attr))
        if hasattr(m, 'maps'):
            maps = getattr(m, 'maps')
            if hasattr(maps, 'tolist'):
                kd['maps'] = [float(v) for v in maps.tolist()]
        if hasattr(m, 'all_ap'):
            ap = getattr(m, 'all_ap')
            if hasattr(ap, 'detach'):
                ap = ap.detach().cpu()
            else:
                ap = torch.as_tensor(ap)
            if ap.ndim == 2 and ap.shape[0] >= 3:
                kd['class_map50'] = [float(v) for v in ap[:, 0].tolist()]
                kd['class_map50_95'] = [float(v) for v in ap.mean(dim=1).tolist()]
        out[kind] = kd
    return out


summaries = []
for run in RUNS:
    print('\n' + '=' * 100)
    print('RUN', run['name'])
    print('WEIGHTS', run['weights'])
    print('DATA', DATA)
    model = YOLO(run['weights'])
    patched = patch_legacy_coordattv2(model.model) if run['legacy'] else 0
    if patched:
        print('patched legacy CoordAttV2 modules:', patched)
    metrics = model.val(
        data=DATA,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        iou=0.7,
        save_txt=False,
        save_json=False,
        plots=False,
        verbose=True,
        stack_metric=True,
        stack_metric_cls=1,
        stack_metric_conf=0.55,
        business_stack_metric=True,
        business_stack2_metric=True,
        business_stack_single_cls=0,
        business_stack_close_mm=12.25,
        business_stack_pix_to_mm=0.35,
        project=PROJECT,
        name=run['name'],
        exist_ok=True,
    )
    s = metric_summary(metrics)
    s['name'] = run['name']
    s['weights'] = run['weights']
    s['data'] = DATA
    s['patched_legacy_coordattv2'] = patched
    summaries.append(s)
    OUT_JSON.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SUMMARY_JSON_PARTIAL', json.dumps(s, ensure_ascii=False, indent=2))

print('\nALL_SUMMARIES_JSON', OUT_JSON)
print(json.dumps(summaries, ensure_ascii=False, indent=2))
