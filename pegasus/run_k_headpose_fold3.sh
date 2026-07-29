#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=12:00:00
#PBS -N K_hp3
#PBS -o logs/pegasus/K_headpose_fold3_out.log
#PBS -e logs/pegasus/K_headpose_fold3_err.log

# Head-pose stream v2 on the minimal frozen form: analytic angles/deltas/
# local-shape features, one token per frame per view. The raw-kpt streams
# measured zero contribution because 60% of their input rows were hands in
# unnormalised identity-bearing coordinates.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
FRAMES="${FRAMES:-32}"
MAX_EPOCHS=50

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "K_headpose_fold3" \
    data.split_mode=person_kfold \
    data.num_folds=5 \
    data.fold=3 \
    train.view=multi "train.view_name=[front,left,right]" model.input_type=rgb_kpt model.backbone=vivit model.fuse_method=mv_vivit model.vivit_model_revision=refs/pr/3 model.mv_vivit_unfreeze_last_layers=0 model.mv_vivit_head_pose_stream=true
