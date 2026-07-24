#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=2:00:00
#PBS -N S0_smoke
#PBS -o logs/pegasus/S0_smoke_single_front_rgb_3dcnn_out.log
#PBS -e logs/pegasus/S0_smoke_single_front_rgb_3dcnn_err.log

# Smoke test: 1 epoch to validate env/data wiring before submitting the matrix.

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "S0_smoke_single_front_rgb_3dcnn" \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=3dcnn
