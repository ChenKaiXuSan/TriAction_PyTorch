#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N N_mv_vivit
#PBS -o logs/pegasus/N_multi_mv_vivit_out.log
#PBS -e logs/pegasus/N_multi_mv_vivit_err.log

# MV-ViViT: frozen shared pretrained ViViT + trainable cross-view token fusion.
# Weights must already be in HF_HOME (pegasus/prepare_hf_models.sh).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Backbone runs under no_grad, only ~15M fusion params train: larger batch fits.
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
FRAMES="${FRAMES:-32}"

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "N_multi_mv_vivit" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.backbone=vivit \
    model.fuse_method=mv_vivit \
    model.vivit_model_revision=refs/pr/3
