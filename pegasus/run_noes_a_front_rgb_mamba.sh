#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N nA_mamba
#PBS -o logs/pegasus/A_front_rgb_mamba_noes_out.log
#PBS -e logs/pegasus/A_front_rgb_mamba_noes_err.log

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "A_front_rgb_mamba_noes" \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=mamba \
    train.early_stopping=false
