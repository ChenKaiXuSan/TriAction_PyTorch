#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-drivefusion}"
GPU_IDS="${GPU_IDS:-[0,1]}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
FOLD="${FOLD:-2}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

run_exp() {
  local exp_id="$1"
  shift

  echo
  echo "===== ${exp_id} ====="
  echo "GPUs=${GPU_IDS} batch_size=${BATCH_SIZE} workers=${NUM_WORKERS} epochs=${MAX_EPOCHS} fold=${FOLD}"

  local cmd=(
    conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" -m project.main
    "train.gpu=${GPU_IDS}" \
    "data.batch_size=${BATCH_SIZE}" \
    "data.num_workers=${NUM_WORKERS}" \
    "train.max_epochs=${MAX_EPOCHS}" \
    "data.fold=${FOLD}" \
    experiment="${exp_id}" \
    "$@"
  )

  if [[ -n "${EXTRA_OVERRIDES}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=(${EXTRA_OVERRIDES})
    cmd+=("${extra_args[@]}")
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi

  "${cmd[@]}"
}

run_smoke() {
  local old_batch_size="${BATCH_SIZE}"
  local old_epochs="${MAX_EPOCHS}"
  local old_fold="${FOLD}"
  BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
  MAX_EPOCHS="${SMOKE_MAX_EPOCHS:-1}"
  FOLD="${SMOKE_FOLD:-1}"
  run_exp "S0_smoke_single_front_rgb_3dcnn" \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=3dcnn
  BATCH_SIZE="${old_batch_size}"
  MAX_EPOCHS="${old_epochs}"
  FOLD="${old_fold}"
}

run_single_view() {
  for view in front left right; do
    run_exp "B_single_${view}_rgb_3dcnn" \
      train.view=single \
      "train.view_name=[${view}]" \
      model.input_type=rgb \
      model.backbone=3dcnn
  done
}

run_modality() {
  local view="${MODALITY_VIEW:-front}"
  run_exp "M_${view}_rgb_3dcnn" \
    train.view=single \
    "train.view_name=[${view}]" \
    model.input_type=rgb \
    model.backbone=3dcnn
  run_exp "M_${view}_kpt" \
    train.view=single \
    "train.view_name=[${view}]" \
    model.input_type=kpt
  run_exp "M_${view}_rgb_kpt_3dcnn" \
    train.view=single \
    "train.view_name=[${view}]" \
    model.input_type=rgb_kpt \
    model.backbone=3dcnn \
    model.modality_fusion=concat
}

run_backbone() {
  local view="${BACKBONE_VIEW:-front}"
  for backbone in 3dcnn transformer mamba; do
    run_exp "A_${view}_rgb_${backbone}" \
      train.view=single \
      "train.view_name=[${view}]" \
      model.input_type=rgb \
      "model.backbone=${backbone}"
  done
}

run_fusion() {
  for fuse_method in add concat avg mid late; do
    run_exp "F_multi_rgb_3dcnn_${fuse_method}" \
      train.view=multi \
      "train.view_name=[front,left,right]" \
      model.input_type=rgb \
      model.backbone=3dcnn \
      "model.fuse_method=${fuse_method}"
  done
}

run_late_backbone() {
  for backbone in 3dcnn transformer mamba; do
    run_exp "L_multi_late_${backbone}" \
      train.view=multi \
      "train.view_name=[front,left,right]" \
      model.input_type=rgb \
      "model.backbone=${backbone}" \
      model.fuse_method=late
  done
}

run_ts_cva() {
  run_exp "T_mid_full_heads4" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=3dcnn \
    model.fuse_method=mid \
    model.ts_cva_use_gated_aggregation=true \
    model.ts_cva_use_view_embedding=true \
    model.ts_cva_num_heads=4
  run_exp "T_mid_no_gated_aggregation" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=3dcnn \
    model.fuse_method=mid \
    model.ts_cva_use_gated_aggregation=false \
    model.ts_cva_use_view_embedding=true \
    model.ts_cva_num_heads=4
  run_exp "T_mid_no_view_embedding" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=3dcnn \
    model.fuse_method=mid \
    model.ts_cva_use_gated_aggregation=true \
    model.ts_cva_use_view_embedding=false \
    model.ts_cva_num_heads=4
  run_exp "T_mid_heads8" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=3dcnn \
    model.fuse_method=mid \
    model.ts_cva_use_gated_aggregation=true \
    model.ts_cva_use_view_embedding=true \
    model.ts_cva_num_heads=8
  run_exp "T_mid_heads2" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=3dcnn \
    model.fuse_method=mid \
    model.ts_cva_use_gated_aggregation=true \
    model.ts_cva_use_view_embedding=true \
    model.ts_cva_num_heads=2
}

case "${STAGE}" in
  smoke)
    run_smoke
    ;;
  single_view)
    run_single_view
    ;;
  modality)
    run_modality
    ;;
  backbone)
    run_backbone
    ;;
  fusion)
    run_fusion
    ;;
  late_backbone)
    run_late_backbone
    ;;
  ts_cva)
    run_ts_cva
    ;;
  all)
    run_smoke
    run_single_view
    run_modality
    run_backbone
    run_fusion
    run_late_backbone
    run_ts_cva
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Usage: $0 [smoke|single_view|modality|backbone|fusion|late_backbone|ts_cva|all]" >&2
    exit 2
    ;;
esac
