#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=08:00:00
#PBS -N nB_sgl_left
#PBS -o logs/pegasus/B_single_left_rgb_3dcnn_noes_out.log
#PBS -e logs/pegasus/B_single_left_rgb_3dcnn_noes_err.log

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "B_single_left_rgb_3dcnn_noes" \
    train.view=single \
    "train.view_name=[left]" \
    model.input_type=rgb \
    model.backbone=3dcnn \
    train.early_stopping=false
