#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-drivefusion}"
GPU_SLOTS="${GPU_SLOTS:-0 1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
PRECISION="${PRECISION:-16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-1}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
FOLD="${FOLD:-2}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
SKIP_DONE="${SKIP_DONE:-0}"

# Run one single-GPU process per slot. With the default "0 1", two worker
# processes are launched and each worker runs its assigned experiments in order.
read -r -a GPU_SLOT_LIST <<< "${GPU_SLOTS}"
if [[ "${#GPU_SLOT_LIST[@]}" -eq 0 ]]; then
  echo "GPU_SLOTS must contain at least one GPU id, for example: GPU_SLOTS='0 1'" >&2
  exit 2
fi

QUEUE_DIR="$(mktemp -d /tmp/triaction_matrix.XXXXXX)"
LOG_ROOT="${LOG_ROOT:-logs/experiment_matrix/$(date +%Y%m%d_%H%M%S)}"
JOB_COUNT=0

cleanup() {
  rm -rf "${QUEUE_DIR}"
}
trap cleanup EXIT

run_exp() {
  local exp_id="$1"
  shift

  if [[ "${SKIP_DONE}" == "1" ]] && find "logs/train/${exp_id}" -mindepth 3 -maxdepth 3 -name metrics.txt -print -quit 2>/dev/null | grep -q .; then
    echo
    echo "===== ${exp_id} ====="
    echo "Skipping completed experiment; found logs/train/${exp_id}/*/*/metrics.txt"
    return 0
  fi

  local slot_idx=$((JOB_COUNT % ${#GPU_SLOT_LIST[@]}))
  local gpu_id="${GPU_SLOT_LIST[$slot_idx]}"
  local queue_file="${QUEUE_DIR}/gpu_${gpu_id}.queue"

  echo
  echo "===== ${exp_id} ====="
  echo "GPU=${gpu_id} batch_size=${BATCH_SIZE} accumulate=${ACCUMULATE_GRAD_BATCHES} precision=${PRECISION} workers=${NUM_WORKERS} val_workers=${VAL_NUM_WORKERS} test_workers=${TEST_NUM_WORKERS} prefetch=${PREFETCH_FACTOR} epochs=${MAX_EPOCHS} fold=${FOLD}"

  local cmd=(
    conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" -m project.main
    "train.gpu=[${gpu_id}]" \
    "data.batch_size=${BATCH_SIZE}" \
    "data.num_workers=${NUM_WORKERS}" \
    "data.val_num_workers=${VAL_NUM_WORKERS}" \
    "data.test_num_workers=${TEST_NUM_WORKERS}" \
    "data.prefetch_factor=${PREFETCH_FACTOR}" \
    "train.max_epochs=${MAX_EPOCHS}" \
    "train.precision=${PRECISION}" \
    "train.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}" \
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
    JOB_COUNT=$((JOB_COUNT + 1))
    return 0
  fi

  printf '%q ' "${cmd[@]}" >> "${queue_file}"
  echo >> "${queue_file}"
  JOB_COUNT=$((JOB_COUNT + 1))
}

dispatch_queues() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  if [[ "${JOB_COUNT}" -eq 0 ]]; then
    echo "No experiments selected."
    return 0
  fi

  mkdir -p "${LOG_ROOT}"
  echo
  echo "Queued ${JOB_COUNT} experiments across ${#GPU_SLOT_LIST[@]} GPU worker(s)."
  echo "Logs: ${LOG_ROOT}"

  local pids=()
  local gpu_id
  for gpu_id in "${GPU_SLOT_LIST[@]}"; do
    local queue_file="${QUEUE_DIR}/gpu_${gpu_id}.queue"
    [[ -s "${queue_file}" ]] || continue

    local log_file="${LOG_ROOT}/gpu_${gpu_id}.log"
    (
      set -euo pipefail
      while IFS= read -r cmdline; do
        echo
        echo "===== GPU ${gpu_id}: ${cmdline} ====="
        eval "${cmdline}"
      done < "${queue_file}"
    ) > "${log_file}" 2>&1 &
    pids+=("$!")
    echo "GPU ${gpu_id} worker started: pid=$! log=${log_file}"
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "At least one GPU worker failed. Check logs under ${LOG_ROOT}." >&2
    return "${status}"
  fi

  echo "All experiment workers finished."
}

run_smoke() {
  local old_batch_size="${BATCH_SIZE}"
  local old_epochs="${MAX_EPOCHS}"
  local old_fold="${FOLD}"
  BATCH_SIZE="${SMOKE_BATCH_SIZE:-4}"
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

dispatch_queues
