#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=08:00:00
#PBS -N nL_videomae
#PBS -o logs/pegasus/L_multi_late_videomae_noes_out.log
#PBS -e logs/pegasus/L_multi_late_videomae_noes_err.log

# HF backbones on H100 80GB: larger per-step batch, effective batch kept at 16
# (batch x accum). Weights must already be in HF_HOME (prepare_hf_models.sh).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
FRAMES="${FRAMES:-16}"

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "L_multi_late_videomae_noes" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=videomae \
    model.fuse_method=late \
    model.fusion_mode=logit_mean \
    train.early_stopping=false
