#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N T_heads8
#PBS -o logs/pegasus/T_mid_heads8_out.log
#PBS -e logs/pegasus/T_mid_heads8_err.log

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "T_mid_heads8" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=3dcnn \
    model.fuse_method=mid \
    model.ts_cva_use_gated_aggregation=true \
    model.ts_cva_use_view_embedding=true \
    model.ts_cva_num_heads=8
