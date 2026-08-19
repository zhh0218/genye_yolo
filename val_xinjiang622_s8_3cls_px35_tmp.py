from ultralytics import YOLO

weight = '/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/xinjiang622_3cls/xinjiang622_s8_3cls_from_s8-seed20260627/weights/best.pt'
data = '/workspace/Datasets/xinjiang_622_yolo11_rgbd_seg_3cls_seed20260627/data_3cls.yaml'
model = YOLO(weight)
metrics = model.val(
    data=data,
    split='val',
    imgsz=640,
    batch=1,
    device=0,
    workers=1,
    verbose=True,
    plots=False,
    project='/workspace/ultralytics-main_for_genye/YOLOv11-RGB-D-coord_attv2-genye/xinjiang622_3cls_eval',
    name='s8_3cls_best_conf055_business_px35_batch1',
    exist_ok=True,
    stack_metric_conf=0.55,
    business_stack_close_mm=12.25,
    business_stack_pix_to_mm=0.35,
)
print('RESULTS_DICT')
for k, v in sorted(metrics.results_dict.items()):
    if 'stack' in k or k.startswith('metrics/mAP') or k.startswith('metrics/precision') or k.startswith('metrics/recall'):
        print(k, v)
