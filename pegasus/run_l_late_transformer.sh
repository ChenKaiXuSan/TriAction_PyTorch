#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N L_transformer
#PBS -o logs/pegasus/L_multi_late_transformer_out.log
#PBS -e logs/pegasus/L_multi_late_transformer_err.log

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "L_multi_late_transformer" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=transformer \
    model.fuse_method=late
