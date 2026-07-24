#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-drivefusion}"
STAGE="${1:-all}"
GPU="${GPU:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
SMOKE_MAX_EPOCHS="${SMOKE_MAX_EPOCHS:-1}"
PRECISION="${PRECISION:-16-mixed}"
INDEX_MAPPING="${INDEX_MAPPING:-/tmp/triaction_index_mapping}"

COMMON_OVERRIDES=(
  "train.gpu=[${GPU}]"
  "train.precision=${PRECISION}"
  "data.batch_size=1"
  "data.num_workers=2"
  "data.val_num_workers=1"
  "data.test_num_workers=1"
  "data.prefetch_factor=1"
  "paths.index_mapping=${INDEX_MAPPING}"
)

run_exp() {
  local exp_id="$1"
  local epochs="$2"
  local frames="$3"
  local accumulate="$4"
  shift 4

  echo
  echo "===== ${exp_id} ====="
  conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" -m project.main \
    "${COMMON_OVERRIDES[@]}" \
    "train.max_epochs=${epochs}" \
    "train.accumulate_grad_batches=${accumulate}" \
    "data.uniform_temporal_subsample_num=${frames}" \
    "experiment=${exp_id}" \
    "$@"
}

mkdir -p "${INDEX_MAPPING}" /tmp/matplotlib
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

run_smoke() {
  # Smoke runs first: if a backbone still has a wiring issue, fail before long jobs.
  run_exp "smoke_front_rgb_videomae" "${SMOKE_MAX_EPOCHS}" 16 16 \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=videomae \
    data.num_workers=0 \
    data.val_num_workers=0 \
    data.test_num_workers=0

  run_exp "smoke_front_rgb_vivit" "${SMOKE_MAX_EPOCHS}" 32 16 \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=vivit \
    model.vivit_model_revision=refs/pr/3 \
    data.num_workers=0 \
    data.val_num_workers=0 \
    data.test_num_workers=0
}

run_formal() {
  run_exp "cmp_single_front_rgb_videomae" "${MAX_EPOCHS}" 16 16 \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=videomae

  run_exp "cmp_single_front_rgb_vivit" "${MAX_EPOCHS}" 32 16 \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=vivit \
    model.vivit_model_revision=refs/pr/3

  run_exp "cmp_late_rgb_videomae" "${MAX_EPOCHS}" 16 16 \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=videomae \
    model.fuse_method=late \
    model.fusion_mode=logit_mean

  run_exp "cmp_late_rgb_vivit" "${MAX_EPOCHS}" 32 16 \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=vivit \
    model.vivit_model_revision=refs/pr/3 \
    model.fuse_method=late \
    model.fusion_mode=logit_mean
}

case "${STAGE}" in
  smoke)
    run_smoke
    ;;
  formal)
    run_formal
    ;;
  all)
    run_smoke
    run_formal
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Usage: $0 [smoke|formal|all]" >&2
    exit 2
    ;;
esac
