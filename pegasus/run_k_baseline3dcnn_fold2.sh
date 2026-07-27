#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=08:00:00
#PBS -N K_b3d2
#PBS -o logs/pegasus/K_baseline3dcnn_fold2_out.log
#PBS -e logs/pegasus/K_baseline3dcnn_fold2_err.log

# Person-wise 5-fold CV: validation persons are unseen in training.
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
FRAMES="${FRAMES:-8}"

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "K_baseline3dcnn_fold2" \
    data.split_mode=person_kfold \
    data.num_folds=5 \
    data.fold=2 \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=rgb \
    model.backbone=3dcnn
