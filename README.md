# TriAction PyTorch

PyTorch Lightning codebase for multi-view driver action recognition. The project
supports RGB, SAM-3D keypoint, and RGB+keypoint inputs, with single-view and
multi-view training routes for front, left, and right camera streams.

## Highlights

- Hydra-based experiment configuration in `configs/config.yaml`.
- Person-wise cross validation with `GroupKFold`.
- Single-view training for RGB, keypoint, and RGB+keypoint inputs.
- Multi-view RGB training with early, mid, and late fusion strategies.
- Backbone options for RGB streams: 3D CNN, temporal Transformer, and Mamba.
- Whole-video loading with chunking support to reduce memory pressure.
- Tests for data loading, model selection, fusion paths, loss weighting, and
  TS-CVA components.

## Repository Layout

```text
configs/                     Hydra experiment configuration
doc/                         Implementation notes and usage guides
examples/                    Small usage examples
project/
  dataloader/                Dataset, annotation, and batching utilities
  models/                    RGB, keypoint, and fusion model definitions
  trainer/                   PyTorch Lightning training modules
  main.py                    Cross-validation training entry point
tests/                       Pytest test suite
```

## Data Layout

The default config expects local data outside the repository:

```text
/workspace/data/multi_view_driver_action/
  label/
    person_01_night_high_h265.json
    person_01_night_low_h265.json
    ...
  index_mapping/
  split_mid_end/mini.json

/workspace/data/videos_split/
  01/
    夜多い/
      front.mp4
      right.mp4
      left.mp4
    夜少ない/
    昼多い/
    昼少ない/

/workspace/data/sam3d_body_results_right/
  01/
    夜多い/
      front/
      right/
      left/
```

Update `paths.*` in `configs/config.yaml` if your data lives elsewhere. The
cross-validation code builds samples from label files named
`person_XX_(day|night)_(high|low)_h265.json` and video folders mapped to
`夜多い`, `夜少ない`, `昼多い`, and `昼少ない`.

## Environment

Create a Python environment with PyTorch, PyTorch Lightning, Hydra, TorchVision,
scikit-learn, NumPy, and the model-specific packages you need. For example:

```bash
conda create -n triaction python=3.10
conda activate triaction
pip install torch torchvision pytorch-lightning hydra-core omegaconf scikit-learn pytest
```

Install CUDA-enabled PyTorch builds according to your GPU driver and CUDA
version.

## Training

Run the default experiment:

```bash
python -m project.main
```

Common Hydra overrides:

```bash
# Single-view RGB 3D CNN on the front camera
python -m project.main train.view=single train.view_name='[front]' model.input_type=rgb model.backbone=3dcnn

# Single-view keypoint model
python -m project.main train.view=single train.view_name='[right]' model.input_type=kpt

# Single-view RGB + keypoint model
python -m project.main train.view=single train.view_name='[front]' model.input_type=rgb_kpt model.backbone=3dcnn

# Multi-view mid-fusion RGB training
python -m project.main train.view=multi train.view_name='[front,left,right]' model.input_type=rgb model.backbone=3dcnn model.fuse_method=mid

# Late fusion with a Transformer backbone
python -m project.main train.view=multi train.view_name='[front,left,right]' model.input_type=rgb model.backbone=transformer model.fuse_method=late
```

Logs and checkpoints are written under `logs/` by default and are intentionally
ignored by Git.

## Configuration Notes

Important options in `configs/config.yaml`:

- `train.view`: `single` or `multi`.
- `train.view_name`: camera list, such as `[front]` or `[front,left,right]`.
- `model.input_type`: `rgb`, `kpt`, or `rgb_kpt`.
- `model.backbone`: `3dcnn`, `transformer`, or `mamba` for RGB streams.
- `model.fuse_method`: `add`, `mul`, `concat`, `avg`, `mid`, or `late`.
- `data.max_video_frames`: chunk size for long videos; lower it if training
  runs out of memory.
- `data.fold`: number of person-wise cross-validation folds.

## Tests

Run the test suite with:

```bash
pytest tests
```

Some tests may require optional model dependencies or local data mocks depending
on the selected test file. Start with focused tests when validating a specific
change:

```bash
pytest tests/test_trainer_selection.py tests/models/test_res_3dcnn.py
```

## Additional Documentation

Detailed notes are available in `doc/`, including:

- `doc/DATASET_USAGE.md`
- `doc/WHOLE_VIDEO_DATASET_GUIDE.md`
- `doc/VIDEO_CHUNKING_GUIDE.md`
- `doc/TS-CVA_README.md`
- `doc/OOM_SOLUTIONS_INDEX.md`

## License

No license file is currently included. Add one before public distribution if the
code will be reused outside the project team.
