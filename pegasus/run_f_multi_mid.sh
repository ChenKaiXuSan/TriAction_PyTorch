#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N F_mid
#PBS -o logs/pegasus/F_multi_rgb_3dcnn_mid_out.log
#PBS -e logs/pegasus/F_multi_rgb_3dcnn_mid_err.log

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "F_multi_rgb_3dcnn_mid" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=3dcnn \
    model.fuse_method=mid
