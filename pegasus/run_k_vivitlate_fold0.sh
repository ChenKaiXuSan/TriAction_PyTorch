#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=12:00:00
#PBS -N K_vl0
#PBS -o logs/pegasus/K_vivitlate_fold0_out.log
#PBS -e logs/pegasus/K_vivitlate_fold0_err.log

# Person-wise 5-fold ablation: three-view ViViT with late logit fusion -- cross-view attention vs trivial fusion
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-16}"
FRAMES="${FRAMES:-32}"
# pinned: every fold in the comparison must train the same number of epochs
MAX_EPOCHS=50

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "K_vivitlate_fold0" \
    data.split_mode=person_kfold \
    data.num_folds=5 \
    data.fold=0 \
    model.backbone=vivit \
    model.vivit_model_revision=refs/pr/3 \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.fuse_method=late \
    model.fusion_mode=logit_mean
