#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/ultralytics-main_for_genye}"
cd "$ROOT"

variant="${1:-}"
if [[ -z "$variant" ]]; then
  cat <<'USAGE'
Usage:
  bash tools/run_genye_teacher_ablation_one.sh <variant>

Variants:
  coord_att_only
  coord_attv2_only
  coord_attv2_p3wa
  coord_attv2_depthfpn
  full
  concat
  add
  old_cmx_ffm
  old_cmx_ffm_stable

Common overrides:
  DEVICE=0 BATCH=16 EPOCHS=300 WORKERS=2 bash tools/run_genye_teacher_ablation_one.sh concat
USAGE
  exit 1
fi

export LOCAL_PROJECT="${LOCAL_PROJECT:-YOLOv11-RGB-D-genye-teacher-ablation}"
export USE_WANDB="${USE_WANDB:-0}"

common_off() {
  export P3_WAVELET_GUIDED=0
  export P3_WAVELET_HF=0
  export P3_WAVELET_ADAPTIVE=0
  export P4_WAVELET_GUIDED=0
  export DEPTH_FPN=0
  export MODALITY_ADAPTIVE_GATE=0
}

case "$variant" in
  coord_att_only)
    common_off
    export RGBD_FUSION=coord_att
    export LOCAL_NAME="${LOCAL_NAME:-YOLO11s-genye-ablation-coordatt}"
    ;;
  coord_attv2_only)
    common_off
    export RGBD_FUSION=coord_att_v2
    export LOCAL_NAME="${LOCAL_NAME:-YOLO11s-genye-ablation-coordattv2}"
    ;;
  coord_attv2_p3wa)
    common_off
    export RGBD_FUSION=coord_att_v2
    export P3_WAVELET_GUIDED=1
    export LOCAL_NAME="${LOCAL_NAME:-YOLO11s-genye-ablation-coordattv2-p3wa}"
    ;;
  coord_attv2_depthfpn)
    common_off
    export RGBD_FUSION=coord_att_v2
    export DEPTH_FPN=1
    export LOCAL_NAME="${LOCAL_NAME:-YOLO11s-genye-ablation-coordattv2-depthfpn}"
    ;;
  full)
    common_off
    export RGBD_FUSION=coord_att_v2
    export P3_WAVELET_GUIDED=1
    export DEPTH_FPN=1
    export LOCAL_NAME="${LOCAL_NAME:-YOLO11s-genye-ablation-coordattv2-p3wa-depthfpn}"
    ;;
  concat)
    common_off
    export RGBD_FUSION=concat
    export LOCAL_NAME="${LOCAL_NAME:-YOLO11s-genye-ablation-concat}"
    ;;
  add)
    common_off
    export RGBD_FUSION=add
    export LOCAL_NAME="${LOCAL_NAME:-YOLO11s-genye-ablation-add}"
    ;;
  old_cmx_ffm)
    common_off
    export RGBD_FUSION=old_cmx_ffm
    export LOCAL_NAME="${LOCAL_NAME:-YOLO11s-genye-ablation-old-cmx-ffm}"
    ;;
  old_cmx_ffm_stable)
    common_off
    export RGBD_FUSION=old_cmx_ffm
    export LOCAL_NAME="${LOCAL_NAME:-YOLO11s-genye-ablation-old-cmx-ffm-stable}"
    ;;
  *)
    echo "Unknown variant: $variant" >&2
    exit 2
    ;;
esac

echo "Running variant=$variant name=$LOCAL_NAME device=${DEVICE:-0,1,2,3} batch=${BATCH:-32} epochs=${EPOCHS:-300}"
python train.py
