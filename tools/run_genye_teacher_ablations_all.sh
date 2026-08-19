#!/usr/bin/env bash
set -euo pipefail

variants=(
  coord_att_only
  coord_attv2_only
  coord_attv2_p3wa
  coord_attv2_depthfpn
  full
  concat
  add
  old_cmx_ffm
)

for variant in "${variants[@]}"; do
  echo "===== $variant ====="
  bash tools/run_genye_teacher_ablation_one.sh "$variant"
done
