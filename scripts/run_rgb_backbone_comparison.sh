#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-drivefusion}"
GPU="${GPU:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
PRECISION="${PRECISION:-16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-1}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
FRAMES="${FRAMES:-8}"
VIEW="${VIEW:-front}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# Conservative defaults for HF video models. Override from the shell if your GPU
# can handle larger batches, for example: HF_BATCH_SIZE=2 HF_ACCUM=8.
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-4}"
BASELINE_ACCUM="${BASELINE_ACCUM:-4}"
HF_BATCH_SIZE="${HF_BATCH_SIZE:-1}"
HF_ACCUM="${HF_ACCUM:-16}"
LATE_BASELINE_BATCH_SIZE="${LATE_BASELINE_BATCH_SIZE:-2}"
LATE_BASELINE_ACCUM="${LATE_BASELINE_ACCUM:-8}"
LATE_HF_BATCH_SIZE="${LATE_HF_BATCH_SIZE:-1}"
LATE_HF_ACCUM="${LATE_HF_ACCUM:-16}"

BASELINE_BACKBONES=(3dcnn transformer mamba)
HF_BACKBONES=(videomae vivit)
ALL_BACKBONES=(3dcnn transformer mamba videomae vivit)

run_exp() {
  local exp_id="$1"
  local batch_size="$2"
  local accumulate="$3"
  shift 3

  local cmd=(
    conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" -m project.main
    "train.gpu=[${GPU}]"
    "train.max_epochs=${MAX_EPOCHS}"
    "train.precision=${PRECISION}"
    "train.accumulate_grad_batches=${accumulate}"
    "data.batch_size=${batch_size}"
    "data.num_workers=${NUM_WORKERS}"
    "data.val_num_workers=${VAL_NUM_WORKERS}"
    "data.test_num_workers=${TEST_NUM_WORKERS}"
    "data.prefetch_factor=${PREFETCH_FACTOR}"
    "data.uniform_temporal_subsample_num=${FRAMES}"
    "experiment=${exp_id}"
    "$@"
  )

  if [[ -n "${EXTRA_OVERRIDES}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=(${EXTRA_OVERRIDES})
    cmd+=("${extra_args[@]}")
  fi

  echo
  echo "===== ${exp_id} ====="
  printf '%q ' "${cmd[@]}"
  echo

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  "${cmd[@]}"
}

batch_for_backbone() {
  local backbone="$1"
  local mode="$2"

  case "${mode}:${backbone}" in
    single:videomae|single:vivit)
      echo "${HF_BATCH_SIZE} ${HF_ACCUM}"
      ;;
    late:videomae|late:vivit)
      echo "${LATE_HF_BATCH_SIZE} ${LATE_HF_ACCUM}"
      ;;
    late:*)
      echo "${LATE_BASELINE_BATCH_SIZE} ${LATE_BASELINE_ACCUM}"
      ;;
    *)
      echo "${BASELINE_BATCH_SIZE} ${BASELINE_ACCUM}"
      ;;
  esac
}

run_smoke() {
  local old_epochs="${MAX_EPOCHS}"
  MAX_EPOCHS="${SMOKE_MAX_EPOCHS:-1}"

  for backbone in "${HF_BACKBONES[@]}"; do
    read -r batch_size accumulate <<< "$(batch_for_backbone "${backbone}" single)"
    run_exp "smoke_${VIEW}_rgb_${backbone}" "${batch_size}" "${accumulate}" \
      train.view=single \
      "train.view_name=[${VIEW}]" \
      model.input_type=rgb \
      "model.backbone=${backbone}" \
      data.num_workers=0 \
      data.val_num_workers=0 \
      data.test_num_workers=0
  done

  MAX_EPOCHS="${old_epochs}"
}

run_single() {
  for backbone in "${ALL_BACKBONES[@]}"; do
    read -r batch_size accumulate <<< "$(batch_for_backbone "${backbone}" single)"
    run_exp "cmp_single_${VIEW}_rgb_${backbone}" "${batch_size}" "${accumulate}" \
      train.view=single \
      "train.view_name=[${VIEW}]" \
      model.input_type=rgb \
      "model.backbone=${backbone}"
  done
}

run_late() {
  for backbone in "${ALL_BACKBONES[@]}"; do
    read -r batch_size accumulate <<< "$(batch_for_backbone "${backbone}" late)"
    run_exp "cmp_late_rgb_${backbone}" "${batch_size}" "${accumulate}" \
      train.view=multi \
      "train.view_name=[front,left,right]" \
      model.input_type=rgb \
      "model.backbone=${backbone}" \
      model.fuse_method=late \
      model.fusion_mode=logit_mean
  done
}

case "${STAGE}" in
  smoke)
    run_smoke
    ;;
  single)
    run_single
    ;;
  late)
    run_late
    ;;
  all)
    run_smoke
    run_single
    run_late
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Usage: $0 [smoke|single|late|all]" >&2
    exit 2
    ;;
esac
