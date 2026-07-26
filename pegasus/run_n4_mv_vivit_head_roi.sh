#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N N4_head_roi
#PBS -o logs/pegasus/N4_mv_vivit_head_roi_out.log
#PBS -e logs/pegasus/N4_mv_vivit_head_roi_err.log

# MV-ViViT round 4: keypoint-guided variants. HF weights via prepare_hf_models.sh.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
FRAMES="${FRAMES:-32}"

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "N4_mv_vivit_head_roi" \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb_kpt \
    model.backbone=vivit \
    model.fuse_method=mv_vivit \
    model.vivit_model_revision=refs/pr/3 \
    model.mv_vivit_unfreeze_last_layers=4 \
    data.head_roi_stream=true \
    model.mv_vivit_head_stream=true
