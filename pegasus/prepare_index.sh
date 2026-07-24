#!/bin/bash
set -euo pipefail

# Pre-generate the train/val split index on the login node BEFORE submitting
# jobs. All experiments share paths.index_mapping; if several jobs start with
# the index missing, they race to regenerate it. Running this once avoids that.
#
# Usage (from anywhere):
#   pegasus/prepare_index.sh
# Override data root or add Hydra overrides:
#   TRIACTION_DATA_ROOT=/path pegasus/prepare_index.sh
#   INDEX_OVERRIDES="data.magic_move=false" pegasus/prepare_index.sh

REPO_ROOT="${TRIACTION_REPO_ROOT:-/work/SKIING/chenkaixu/code/TriAction_PyTorch}"
export DATA_ROOT="${TRIACTION_DATA_ROOT:-/work/SKIING/chenkaixu/data/drive}"

cd "${REPO_ROOT}"

source pegasus/setup_env.sh

python - <<'PY'
import os

from hydra import compose, initialize_config_dir

from project.cross_validation import DefineCrossValidation

data_root = os.environ["DATA_ROOT"]
overrides = [
    f"paths.root_path={data_root}",
    f"paths.video_path={data_root}/videos_split",
    f"paths.sam3d_results_path={data_root}/sam3d_body_results_right",
    f"paths.start_mid_end_path={data_root}/annotation/split_mid_end/mini.json",
]
overrides += os.environ.get("INDEX_OVERRIDES", "").split()

with initialize_config_dir(
    config_dir=os.path.join(os.getcwd(), "configs"), version_base=None
):
    config = compose(config_name="config", overrides=overrides)

split = DefineCrossValidation(config)()
print(f"index ready at {config.paths.index_mapping}")
print(f"train samples: {len(split['train'])}, val samples: {len(split['val'])}")
PY
