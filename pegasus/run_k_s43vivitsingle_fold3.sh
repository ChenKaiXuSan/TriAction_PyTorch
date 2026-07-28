#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=08:00:00
#PBS -N K_s43v3
#PBS -o logs/pegasus/K_s43vivitsingle_fold3_out.log
#PBS -e logs/pegasus/K_s43vivitsingle_fold3_err.log

# seed 43 replicate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
FRAMES="${FRAMES:-32}"
MAX_EPOCHS=50

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "K_s43vivitsingle_fold3" \
    data.split_mode=person_kfold \
    data.num_folds=5 \
    data.fold=3 \
    train.view=single "train.view_name=[front]" model.input_type=rgb model.backbone=vivit model.vivit_model_revision=refs/pr/3 train.seed=43
