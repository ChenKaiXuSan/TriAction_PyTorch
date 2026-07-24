#!/bin/bash
set -euo pipefail

# Prepare Hugging Face backbones (videomae / vivit) on the LOGIN node, which
# has internet access. Compute nodes run offline (HF_HUB_OFFLINE=1), so the
# weights must already sit in the shared HF_HOME cache before submitting
# run_a_front_rgb_videomae.sh / run_a_front_rgb_vivit.sh / run_l_late_*.sh.
#
# Does three things:
#   1. pip install transformers into the conda env if it is missing.
#   2. snapshot_download the videomae and vivit checkpoints into HF_HOME.
#   3. cache the SlowR50 Kinetics weights for the 3dcnn backbone (also loaded
#      offline on compute nodes, from ~/.cache/pytorchvideo/slow_r50.pyth).

REPO_ROOT="${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}"

cd "${REPO_ROOT}"

source pegasus/setup_env.sh

if ! python -c 'import transformers' >/dev/null 2>&1; then
    echo "transformers not found in $(which python); installing..."
    python -m pip install transformers
fi

# SlowR50 (3dcnn backbone): project/models/base_model.py falls back to random
# init when this cache is missing and there is no internet.
SLOW_R50_CACHE="${HOME}/.cache/pytorchvideo/slow_r50.pyth"
if [[ ! -f "${SLOW_R50_CACHE}" ]]; then
    mkdir -p "$(dirname "${SLOW_R50_CACHE}")"
    echo "downloading SlowR50 Kinetics weights..."
    curl -fL -o "${SLOW_R50_CACHE}" \
        https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth
fi
echo "SlowR50 weights: ${SLOW_R50_CACHE}"

python - <<'PY'
from huggingface_hub import snapshot_download

for repo_id, revision in [
    ("MCG-NJU/videomae-base-finetuned-kinetics", None),
    ("google/vivit-b-16x2-kinetics400", None),
    # The vivit jobs pin model.vivit_model_revision=refs/pr/3 (safetensors weights).
    ("google/vivit-b-16x2-kinetics400", "refs/pr/3"),
]:
    print(f"downloading {repo_id} (revision={revision or 'main'}) ...")
    path = snapshot_download(repo_id, revision=revision)
    print(f"  -> {path}")

print("All HF backbones cached.")
PY
