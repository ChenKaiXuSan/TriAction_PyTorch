#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N nM_rgb_kpt
#PBS -o logs/pegasus/M_front_rgb_kpt_3dcnn_noes_out.log
#PBS -e logs/pegasus/M_front_rgb_kpt_3dcnn_noes_err.log

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "M_front_rgb_kpt_3dcnn_noes" \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb_kpt \
    model.backbone=3dcnn \
    model.modality_fusion=concat \
    train.early_stopping=false
