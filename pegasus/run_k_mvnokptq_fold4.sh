#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=12:00:00
#PBS -N K_mnq4
#PBS -o logs/pegasus/K_mvnokptq_fold4_out.log
#PBS -e logs/pegasus/K_mvnokptq_fold4_err.log

# Person-wise 5-fold ablation: MV-ViViT without keypoint guidance -- isolates the kpt query pooling
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
FRAMES="${FRAMES:-32}"
# pinned: every fold in the comparison must train the same number of epochs
MAX_EPOCHS=50

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "K_mvnokptq_fold4" \
    data.split_mode=person_kfold \
    data.num_folds=5 \
    data.fold=4 \
    model.backbone=vivit \
    model.vivit_model_revision=refs/pr/3 \
    train.view=multi \
    "train.view_name=[front,left,right]" \
    model.input_type=rgb \
    model.fuse_method=mv_vivit \
    model.mv_vivit_unfreeze_last_layers=4
