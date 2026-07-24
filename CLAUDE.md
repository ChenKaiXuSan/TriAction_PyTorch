# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch Lightning codebase for multi-view driver action recognition (4-class direction task: left/right/down/up). Supports RGB, SAM-3D keypoint, and RGB+keypoint inputs across front/left/right camera views, with single-view and multi-view (early/mid/late fusion) training routes. Configuration is Hydra-based with a single config file: `configs/config.yaml`.

Comments and docs are mixed English/Chinese; data environment folder names are Japanese (`夜多い`, `夜少ない`, `昼多い`, `昼少ない`) — preserve them exactly.

## Commands

The working conda env is `drivefusion` (torch + pytorch-lightning + hydra). pytest is not preinstalled there — `conda run -n drivefusion pip install pytest` first if needed.

```bash
# Run tests (from repo root — tests import `project.*`, so use python -m so cwd is on sys.path)
conda run -n drivefusion python -m pytest tests

# Single test file / single test
conda run -n drivefusion python -m pytest tests/test_trainer_selection.py
conda run -n drivefusion python -m pytest tests/test_trainer_selection.py -k test_select_multi_trainer_early_fusion

# Train with default config
python -m project.main

# Hydra overrides (examples)
python -m project.main train.view=single train.view_name='[front]' model.input_type=rgb model.backbone=3dcnn
python -m project.main train.view=multi train.view_name='[front,left,right]' model.input_type=rgb model.backbone=3dcnn model.fuse_method=mid
python -m project.main train.view=single train.view_name='[right]' model.input_type=kpt

# Evaluate best checkpoint of a run
python -m project.eval

# Batch experiment sweeps (use `conda run -n drivefusion`; override via env vars like GPU_SLOTS, MAX_EPOCHS, EXTRA_OVERRIDES)
scripts/run_experiment_matrix.sh
scripts/run_rgb_backbone_comparison.sh
scripts/run_missing_hf_backbone_comparison.sh
```

Requires GPU for training (`accelerator="gpu"`); `train.gpu` takes a list of device ids (e.g. `train.gpu=[0]`). Model tests (e.g. `tests/models/test_hf_video_backbones.py`) may need optional deps (`transformers`, `mamba`); most dataloader/trainer tests run CPU-only with mocks.

## Architecture

### Training flow (project/main.py)

1. `DefineCrossValidation(config)()` (project/cross_validation.py) scans `paths.annotation_path` for label files named `person_XX_(day|night)_(high|low)_h265.json`, pairs each with videos at `{video_path}/{person}/{env_folder}/{front,left,right}.mp4` and optional SAM-3D keypoint dirs, producing a **single train/val split** (K-fold was removed; `normalize_dataset_split` still accepts the legacy one-fold mapping shape). With `data.magic_move: true`, a ratio of train samples is moved to val.
   - **The split is cached as JSON** in `paths.index_mapping` (`index_single.json` / `index_single_magicmove.json`). It is only regenerated if the file is missing — delete it after changing data paths or split logic.
2. A trainer (LightningModule) is selected by `train.view`:
   - `single` → `project/trainer/single_selector.py` → `SingleModalityClassifierTrainer` (handles all of rgb/kpt/rgb_kpt; wraps `select_model` from project/models/make_model.py).
   - `multi` → `project/trainer/multi_selector.py`, routed by `model.fuse_method`: early (`add|mul|concat|avg`), mid (`mid` → `MultiTSCVATrainer`, 3dcnn backbone only), late (`late`, per-backbone trainer classes). Multi-view only supports `model.input_type=rgb`.
3. `DriverDataModule` (project/dataloader/data_loader.py) + PL `Trainer` run `fit`, then `test` with `ckpt_path="best"` (checkpoint/early-stop monitor `val/loss`). Logs, checkpoints, TensorBoard, and CSV metrics go to `logs/train/${experiment}/<date>/<time>`.

`project/eval.py` mirrors this flow but locates the best checkpoint under `log_path` and only runs `trainer.test`.

### Data pipeline

- `project/map_config.py` is the shared vocabulary: `VideoSample` dataclass, label id↔name mapping, the 8→4 class merge (`normalize_label_to_4_class`, vertical wins: `left_up`→`up`), env-key→Japanese-folder mapping, `CAM_NAMES`, and `KEEP_KEYPOINT_INDICES` (head/shoulders/hands subset of SAM-3D keypoints).
- `LabeledVideoDataset` (project/dataloader/whole_video_dataset.py, ~1200 lines) loads whole videos and splits them into labeled segments using the annotation timeline:
  - Long videos are chunked via `data.max_video_frames` to avoid OOM; `data.batch_unit: segment|chunk` decides whether `__getitem__` yields one labeled segment (making `data.batch_size` the true batch size) or one whole chunk.
  - Has FPS/label/frame LRU caches; `SegmentGroupedSampler` in data_loader.py shuffles chunk groups while keeping same-chunk segments adjacent so the frame cache is reused.
  - Unlabeled gaps are skipped (no background class in the 4-class task).
- Batch dict shape: `batch["video"][view]` → (B, T, C, H, W), `batch["kpt"][view]` → keypoints, `batch["label"]` → (B,). Trainers pull the views listed in `train.view_name`.

### Models

`select_model` (project/models/make_model.py) routes on `model.input_type` (rgb / kpt / rgb_kpt); RGB backbones are lazily imported on `model.backbone`: `3dcnn`, `transformer`, `mamba`, `videomae`, `vivit`. VideoMAE/ViViT live in `project/models/hf_video_backbone.py` and are Hugging Face models; `model.hf_video_pretrained: false` gives random init for offline/local debugging. TS-CVA (mid fusion) is in `project/models/ts_cva_model.py` with `ts_cva_*` config knobs.

Loss helpers in `project/trainer/losses.py` build per-class weights from `loss.class_weights` (keyed by label *names*, ordered via `label_mapping_Dict`).

`project/trainer/legacy/` contains old trainers that are not routed by the selectors — don't extend them.

### Notes

- Adding a new backbone or fusion method means touching both the selector maps (`single_selector.py` / `multi_selector.py`) and `make_model.py`, plus `tests/test_trainer_selection.py`.
- Default data paths in `configs/config.yaml` point to machine-local dirs (`/home/data/xchen/...`); override `paths.*` rather than editing them for experiments.
- `doc/` holds detailed guides (dataset usage, video chunking, OOM fixes, TS-CVA); `docs/superpowers/` holds design specs/plans.
