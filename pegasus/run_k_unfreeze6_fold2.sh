#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=12:00:00
#PBS -N K_u62
#PBS -o logs/pegasus/K_unfreeze6_fold2_out.log
#PBS -e logs/pegasus/K_unfreeze6_fold2_err.log

# unfreeze depth 6 (claimed best under the old protocol)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
FRAMES="${FRAMES:-32}"
MAX_EPOCHS=50

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "K_unfreeze6_fold2" \
    data.split_mode=person_kfold \
    data.num_folds=5 \
    data.fold=2 \
    train.view=multi "train.view_name=[front,left,right]" model.input_type=rgb_kpt model.backbone=vivit model.fuse_method=mv_vivit model.vivit_model_revision=refs/pr/3 model.mv_vivit_unfreeze_last_layers=4 model.mv_vivit_kpt_stream=true model.mv_vivit_kpt_query_pooling=true model.mv_vivit_unfreeze_last_layers=6
