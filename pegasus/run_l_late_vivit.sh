#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N L_vivit
#PBS -o logs/pegasus/L_multi_late_vivit_out.log
#PBS -e logs/pegasus/L_multi_late_vivit_err.log

# HF backbones: big models, small per-step batch; weights must already be in
# HF_HOME (run pegasus/prepare_hf_models.sh on the login node first).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-16}"
FRAMES="${FRAMES:-32}"

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "L_multi_late_vivit" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=vivit \
    model.vivit_model_revision=refs/pr/3 \
    model.fuse_method=late \
    model.fusion_mode=logit_mean
