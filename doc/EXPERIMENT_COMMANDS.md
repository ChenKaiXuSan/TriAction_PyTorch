# Experiment Commands

Use module execution from the repository root:

```bash
conda activate drivefusion
python -m project.main
```

Do not use `python project/main.py`; running the file path directly can break
package imports.

## Path Template

On this machine, the dataset is laid out as:

```bash
DATA_ROOT=/home/data/xchen/drive/multi_view_driver_action
VIDEO_ROOT=/home/data/xchen/drive/videos_split
KPT_ROOT=/home/data/xchen/drive/sam3d_body_results_right
SPLIT_JSON=$DATA_ROOT/split_mid_end/mini.json
```

Append the following overrides to every command because the default config still
uses `/workspace/data/...` paths:

```bash
paths.root_path=$DATA_ROOT \
paths.annotation_path=$DATA_ROOT/label \
paths.index_mapping=$DATA_ROOT/index_mapping \
paths.start_mid_end_path=$SPLIT_JSON \
paths.video_path=$VIDEO_ROOT \
paths.sam3d_results_path=$KPT_ROOT
```

If you are running from a restricted environment where `/home/data/...` is
read-only, keep the data paths above but write generated split indices and logs
to a writable local folder:

```bash
paths.index_mapping=/tmp/triaction_index_mapping \
log_path=/tmp/triaction_logs/${experiment}
```

Useful lightweight overrides while checking a new setup:

```bash
train.max_epochs=1 data.fold=1 data.num_workers=0 data.batch_size=1
```

## 0. Smoke Runs

The experiment matrix script runs the same commands with dual GPUs and a larger
default batch size:

```bash
# Defaults: train.gpu=[0,1], data.batch_size=2, data.num_workers=12
scripts/run_experiment_matrix.sh smoke

# Print commands without running them
DRY_RUN=1 scripts/run_experiment_matrix.sh all

# Override defaults
BATCH_SIZE=4 MAX_EPOCHS=80 scripts/run_experiment_matrix.sh fusion
GPU_IDS='[0]' BATCH_SIZE=1 scripts/run_experiment_matrix.sh smoke
```

Supported stages are `smoke`, `single_view`, `modality`, `backbone`, `fusion`,
`late_backbone`, `ts_cva`, and `all`.

Single-view RGB smoke run:

```bash
python -m project.main \
  train.view=single \
  train.view_name='[front]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  train.max_epochs=1 \
  data.fold=1 \
  data.num_workers=0 \
  data.batch_size=1
```

Multi-view smoke run:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  model.fuse_method=mid \
  train.max_epochs=1 \
  data.fold=1 \
  data.num_workers=0 \
  data.batch_size=1
```

## 1. Single-View RGB Backbones

Front view, 3D CNN:

```bash
python -m project.main \
  train.view=single \
  train.view_name='[front]' \
  model.input_type=rgb \
  model.backbone=3dcnn
```

Front view, temporal Transformer:

```bash
python -m project.main \
  train.view=single \
  train.view_name='[front]' \
  model.input_type=rgb \
  model.backbone=transformer
```

Front view, Mamba:

```bash
python -m project.main \
  train.view=single \
  train.view_name='[front]' \
  model.input_type=rgb \
  model.backbone=mamba
```

Repeat the same commands with `train.view_name='[left]'` and
`train.view_name='[right]'` for camera-view comparison.

## 2. Single-View Input Modalities

RGB only:

```bash
python -m project.main \
  train.view=single \
  train.view_name='[front]' \
  model.input_type=rgb \
  model.backbone=3dcnn
```

SAM-3D keypoints only:

```bash
python -m project.main \
  train.view=single \
  train.view_name='[front]' \
  model.input_type=kpt
```

RGB + SAM-3D keypoints:

```bash
python -m project.main \
  train.view=single \
  train.view_name='[front]' \
  model.input_type=rgb_kpt \
  model.backbone=3dcnn \
  model.modality_fusion=concat
```

## 3. Multi-View Fusion Strategies

Early fusion with additive fusion:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  model.fuse_method=add
```

Early fusion with concatenation:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  model.fuse_method=concat
```

Mid fusion with TS-CVA-style trainer:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  model.fuse_method=mid
```

Late fusion with 3D CNN:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  model.fuse_method=late \
  model.fusion_mode=logit_mean
```

## 4. Late-Fusion Backbone Comparison

3D CNN:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  model.fuse_method=late
```

Temporal Transformer:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=transformer \
  model.fuse_method=late
```

Mamba:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=mamba \
  model.fuse_method=late
```

## 5. TS-CVA Ablations

Default mid-fusion run:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  model.fuse_method=mid
```

Without gated aggregation:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  model.fuse_method=mid \
  model.ts_cva_use_gated_aggregation=false
```

Without view embeddings:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  model.fuse_method=mid \
  model.ts_cva_use_view_embedding=false
```

Different attention head count:

```bash
python -m project.main \
  train.view=multi \
  train.view_name='[front,left,right]' \
  model.input_type=rgb \
  model.backbone=3dcnn \
  model.fuse_method=mid \
  model.ts_cva_num_heads=8
```

## Suggested Experiment Order

1. Run one single-view smoke test and one multi-view smoke test.
2. Train single-view RGB baselines for `front`, `left`, and `right`.
3. Compare single-view `rgb`, `kpt`, and `rgb_kpt` on the best view.
4. Compare multi-view `add`, `concat`, `mid`, and `late` with 3D CNN.
5. Compare late-fusion backbones: 3D CNN, Transformer, and Mamba.
6. Run TS-CVA ablations against the best multi-view baseline.
