#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=08:00:00
#PBS -N K_ts0
#PBS -o logs/pegasus/K_tscva_fold0_out.log
#PBS -e logs/pegasus/K_tscva_fold0_err.log

# TS-CVA mid fusion on 3dcnn -- the old mid-fusion champion
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
FRAMES="${FRAMES:-32}"
MAX_EPOCHS=50

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "K_tscva_fold0" \
    data.split_mode=person_kfold \
    data.num_folds=5 \
    data.fold=0 \
    train.view=multi "train.view_name=[front,left,right]" model.input_type=rgb model.backbone=3dcnn model.fuse_method=mid
