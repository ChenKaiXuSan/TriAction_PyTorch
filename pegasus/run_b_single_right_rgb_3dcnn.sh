#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N B_sgl_right
#PBS -o logs/pegasus/B_single_right_rgb_3dcnn_out.log
#PBS -e logs/pegasus/B_single_right_rgb_3dcnn_err.log

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "B_single_right_rgb_3dcnn" \
    train.view=single \
    "train.view_name=[right]" \
    model.input_type=rgb \
    model.backbone=3dcnn
