#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N E_noes_front
#PBS -o logs/pegasus/E_noes_single_front_rgb_3dcnn_out.log
#PBS -e logs/pegasus/E_noes_single_front_rgb_3dcnn_err.log

# Reference run without early stopping: the matrix runs stop at ~8-9 epochs on
# the val/loss plateau; this one trains the full max_epochs for comparison.

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "E_noes_single_front_rgb_3dcnn" \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=3dcnn \
    train.early_stopping=false
