#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${RELEASE_ROOT}"
PY="${PY:-/root/miniconda3/envs/yolo26/bin/python}"
MODEL_CFG="${RELEASE_ROOT}/configs/yolo11-seg-spatialillum-learnblend-auxoff-xinjiang1500.yaml"
PRETRAIN="${RELEASE_ROOT}/YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt"
RUN_ROOT="${RELEASE_ROOT}/YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments_release_rerun"
mkdir -p "${RUN_ROOT}"
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-12}"
VAL_PROJECT="${RELEASE_ROOT}/YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_eval_release"
mkdir -p "${VAL_PROJECT}"

run_val() {
  local name="$1"
  local weight="$2"
  local data="$3"
  "${PY}" -c "from ultralytics.cfg import entrypoint; raise SystemExit(entrypoint())" segment val \
    model="${weight}" \
    data="${data}" \
    imgsz=640 \
    batch="${BATCH}" \
    device="${DEVICE}" \
    workers=2 \
    project="${VAL_PROJECT}" \
    name="${name}" \
    exist_ok=True \
    split=val \
    stack_metric=True \
    stack_metric_cls=1 \
    stack_metric_conf=0.55 \
    business_stack_metric=True \
    business_stack2_metric=True \
    business_stack_single_cls=0 \
    business_stack_close_mm=12.25 \
    business_stack_pix_to_mm=0.35 \
    iou=0.7 \
    max_det=300 \
    plots=True
}

run_val "clean_best_on_clean_val" \
  "${RELEASE_ROOT}/YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/xinjiang_1500_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt" \
  "${RELEASE_ROOT}/Dataset/xinjiang_1500/data_3cls.yaml"

run_val "baoguang_best_on_clean_val" \
  "${RELEASE_ROOT}/YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/xinjiang_1500_baoguang_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt" \
  "${RELEASE_ROOT}/Dataset/xinjiang_1500_baoguang/data_3cls.yaml"

run_val "baoguang_best_on_val_expaug" \
  "${RELEASE_ROOT}/YOLOv11-RGB-D-coord_attv2-genye/xinjiang_1500_experiments/xinjiang_1500_baoguang_from_coordattv2s8_lr001-spillum-learnblend-auxoff-3gpu_b36-seed20260702/weights/best.pt" \
  "${RELEASE_ROOT}/Dataset/xinjiang_1500_baoguang/data_3cls_val_expaug.yaml"
