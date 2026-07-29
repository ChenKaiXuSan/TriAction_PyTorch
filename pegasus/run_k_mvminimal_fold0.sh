#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=12:00:00
#PBS -N K_min0
#PBS -o logs/pegasus/K_mvminimal_fold0_out.log
#PBS -e logs/pegasus/K_mvminimal_fold0_err.log

# The method in its minimal form: frozen shared ViViT + cross-view token
# attention only. Ablations showed keypoint guidance and partial unfreezing
# contribute nothing, so this is what the paper should call "our method".
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
FRAMES="${FRAMES:-32}"
MAX_EPOCHS=50

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "K_mvminimal_fold0" \
    data.split_mode=person_kfold \
    data.num_folds=5 \
    data.fold=0 \
    train.view=multi "train.view_name=[front,left,right]" model.input_type=rgb model.backbone=vivit model.fuse_method=mv_vivit model.vivit_model_revision=refs/pr/3 model.mv_vivit_unfreeze_last_layers=0
