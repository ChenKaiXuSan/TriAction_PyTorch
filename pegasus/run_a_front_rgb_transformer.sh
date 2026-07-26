#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=08:00:00
#PBS -N A_transformer
#PBS -o logs/pegasus/A_front_rgb_transformer_out.log
#PBS -e logs/pegasus/A_front_rgb_transformer_err.log

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "A_front_rgb_transformer" \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=transformer
