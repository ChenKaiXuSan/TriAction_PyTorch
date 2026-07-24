#!/bin/bash

# Shared runner for TriAction Pegasus jobs. A job script sets its own knobs
# (BATCH_SIZE, FRAMES, ...) before sourcing this file, then calls run_exp with
# the experiment id and Hydra overrides. All knobs can also be overridden from
# qsub, e.g.: MAX_EPOCHS=30 qsub pegasus/run_f_multi_mid.sh

set -euo pipefail

REPO_ROOT="${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}"
DATA_ROOT="${TRIACTION_DATA_ROOT:-/work/SKIING/chenkaixu/data/drive}"

cd "${REPO_ROOT}"

mkdir -p logs/pegasus/

source pegasus/setup_env.sh

echo "Current working directory: $(pwd)"
echo "Total CPU cores: $(nproc)"

BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
PRECISION="${PRECISION:-16-mixed}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
FRAMES="${FRAMES:-8}"
# Video decode is the training bottleneck; one job owns the whole node.
NUM_WORKERS="${NUM_WORKERS:-$(( $(nproc) / 3 ))}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

run_exp() {
    local exp_id="$1"
    shift

    local cmd=(
        python -m project.main
        "paths.root_path=${DATA_ROOT}"
        "paths.video_path=${DATA_ROOT}/videos_split"
        "paths.sam3d_results_path=${DATA_ROOT}/sam3d_body_results_right"
        "paths.start_mid_end_path=${DATA_ROOT}/annotation/split_mid_end/mini.json"
        "train.gpu=[0]"
        "train.max_epochs=${MAX_EPOCHS}"
        "train.precision=${PRECISION}"
        "train.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
        "data.batch_size=${BATCH_SIZE}"
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

    "${cmd[@]}"
}
