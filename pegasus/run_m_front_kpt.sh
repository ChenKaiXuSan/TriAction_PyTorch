#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=08:00:00
#PBS -N M_front_kpt
#PBS -o logs/pegasus/M_front_kpt_out.log
#PBS -e logs/pegasus/M_front_kpt_err.log

# Keypoint-only training reads small npz files, not video: fewer workers needed.
NUM_WORKERS="${NUM_WORKERS:-4}"

source "${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}/pegasus/run_common.sh"

run_exp "M_front_kpt" \
    train.view=single \
    "train.view_name=[front]" \
    model.input_type=kpt
