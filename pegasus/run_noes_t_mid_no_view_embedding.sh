#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=08:00:00
#PBS -N nT_no_viewemb
#PBS -o logs/pegasus/T_mid_no_view_embedding_noes_out.log
#PBS -e logs/pegasus/T_mid_no_view_embedding_noes_err.log

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "T_mid_no_view_embedding_noes" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=3dcnn \
    model.fuse_method=mid \
    model.ts_cva_use_gated_aggregation=true \
    model.ts_cva_use_view_embedding=false \
    model.ts_cva_num_heads=4 \
    train.early_stopping=false
