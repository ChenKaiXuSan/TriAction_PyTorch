#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=08:00:00
#PBS -N K_vs4
#PBS -o logs/pegasus/K_vivitsingle_fold4_out.log
#PBS -e logs/pegasus/K_vivitsingle_fold4_err.log

# Person-wise 5-fold ablation: single-view ViViT -- isolates the backbone from views and fusion
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
FRAMES="${FRAMES:-32}"
# pinned: every fold in the comparison must train the same number of epochs
MAX_EPOCHS=50

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "K_vivitsingle_fold4" \
    data.split_mode=person_kfold \
    data.num_folds=5 \
    data.fold=4 \
    model.backbone=vivit \
    model.vivit_model_revision=refs/pr/3 \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb
