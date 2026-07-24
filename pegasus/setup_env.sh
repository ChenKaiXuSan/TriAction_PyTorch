#!/bin/bash

# Shared Pegasus environment bootstrap for TriAction training jobs.
# Override when needed:
#   TRIACTION_CONDA_ENV=/path/to/env qsub pegasus/run_b_single_front_rgb_3dcnn.sh

TRIACTION_CONDA_ENV="${TRIACTION_CONDA_ENV:-/home/SKIING/chenkaixu/miniconda3/envs/direction}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-user}-${PBS_JOBID:-manual}}"
# Shared Hugging Face cache (videomae/vivit weights are pre-downloaded here by
# pegasus/prepare_hf_models.sh; compute nodes have no internet access).
export HF_HOME="${HF_HOME:-/work/SKIING/chenkaixu/hf_cache}"

mkdir -p "${MPLCONFIGDIR}"

if command -v module >/dev/null 2>&1; then
    module load intelpython/2022.3.1
fi

if [[ -z "${CONDA_PREFIX:-}" || ! -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: Could not find conda.sh after loading intelpython/2022.3.1" >&2
    echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}" >&2
    exit 1
fi

# conda's shell functions reference unbound variables; suspend `set -u` from
# callers (run_common.sh) while sourcing/activating.
case $- in *u*) _restore_nounset=1 ;; *) _restore_nounset=0 ;; esac
set +u

source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda deactivate >/dev/null 2>&1 || true

if [[ ! -x "${TRIACTION_CONDA_ENV}/bin/python" ]]; then
    echo "ERROR: Conda environment has no executable python: ${TRIACTION_CONDA_ENV}" >&2
    echo "Available conda environments:" >&2
    conda info --envs >&2 || true
    exit 1
fi

conda activate "${TRIACTION_CONDA_ENV}" || {
    echo "ERROR: Failed to activate conda environment: ${TRIACTION_CONDA_ENV}" >&2
    conda info --envs >&2 || true
    exit 1
}

if [[ "${_restore_nounset}" == "1" ]]; then
    set -u
fi
unset _restore_nounset

hash -r

echo "Python version: $(python --version)"
echo "Python executable: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi: not found"
fi

python -c 'import torch; print("torch:", torch.__version__); print("torch cuda available:", torch.cuda.is_available()); print("torch cuda device count:", torch.cuda.device_count())'
