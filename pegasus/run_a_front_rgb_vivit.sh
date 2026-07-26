#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=08:00:00
#PBS -N A_vivit
#PBS -o logs/pegasus/A_front_rgb_vivit_out.log
#PBS -e logs/pegasus/A_front_rgb_vivit_err.log

# HF backbones on H100 80GB: larger per-step batch, effective batch kept at 16
# (batch x accum). Weights must already be in HF_HOME (prepare_hf_models.sh).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
FRAMES="${FRAMES:-32}"

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "A_front_rgb_vivit" \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=vivit \
    model.vivit_model_revision=refs/pr/3
