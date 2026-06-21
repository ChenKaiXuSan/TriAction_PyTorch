# Multiview Driver Action Classification Redesign

Date: 2026-06-21
Project root: /home/workspace/kaixu/code/multiview_drive_action_recognition

## Goal

Refactor the existing legacy code collection into a complete standalone project for multiview driver action classification. The project focuses only on action classification and supports configurable input modalities:

- RGB video only
- Keypoint sequence only
- RGB video and keypoint sequence together

The result should be a clean research and development project that a new user can install, configure, train, evaluate, and test without understanding the old project package layout.

## Non-Goals

The project will not keep CAM visualization, paper-submission notes, ad hoc experiment fragments, or legacy trainer compatibility as first-class APIs. Useful implementation ideas from the legacy code may be ported, but old module names and selector patterns do not need to remain stable.

## Proposed Structure

```text
multiview_drive_action_recognition/
  README.md
  pyproject.toml
  configs/
    train_rgb.yaml
    train_kpt.yaml
    train_rgb_kpt.yaml
  mvda/
    __init__.py
    data/
      __init__.py
      datamodule.py
      datasets.py
      schemas.py
      transforms.py
    models/
      __init__.py
      action_classifier.py
      fusion.py
      kpt_backbones.py
      rgb_backbones.py
    trainers/
      __init__.py
      lightning_module.py
      metrics.py
    engine/
      __init__.py
      cross_validation.py
      evaluate.py
      train.py
    utils/
      __init__.py
      config.py
      paths.py
      seed.py
  scripts/
    train.py
    evaluate.py
  tests/
  docs/
```

## Configuration

Configuration will control the task, dataset paths, views, modalities, model, training, and evaluation. The modality switch will be explicit:

```yaml
data:
  views: [front, left, right]
  modalities: [rgb]
  # modalities: [kpt]
  # modalities: [rgb, kpt]
```

The three starter configs will cover RGB-only, keypoint-only, and RGB+keypoint training. Dataset paths will stay configurable and will not be hard-coded into model or trainer code.

## Data Interface

All datasets and dataloaders will return a consistent batch shape, regardless of the selected modalities:

```python
{
    "rgb": {"front": tensor, "left": tensor, "right": tensor} | None,
    "kpt": {"front": tensor, "left": tensor, "right": tensor} | None,
    "label": tensor,
    "meta": {...},
}
```

If a modality is disabled, its value is None. If a view is not configured, it is absent from that modality dictionary. This keeps the trainer and model code independent of dataset-specific loading details.

The data layer will preserve the useful legacy behavior for whole-video loading, temporal subsampling, chunking long videos to reduce memory pressure, and K-fold split support.

## Model Architecture

The new model API will center on one action classification model:

```text
RGB only:    RGBEncoder -> ViewFusion -> Classifier
KPT only:    KptEncoder -> ViewFusion -> Classifier
RGB + KPT:   RGBEncoder + KptEncoder -> ModalityFusion -> Classifier
```

Initial backbones:

- RGB: 3D CNN baseline ported from the legacy implementation
- KPT: temporal MLP or lightweight temporal encoder baseline
- Fusion: mean, logit, or feature fusion first, with room for TS-CVA-style fusion once the clean baseline is stable

The model should expose a single forward API that accepts the normalized batch dictionaries from the data layer.

## Training And Evaluation

PyTorch Lightning will remain the training framework, but the public entrypoints will be simplified:

```bash
python scripts/train.py --config configs/train_rgb.yaml
python scripts/train.py --config configs/train_kpt.yaml
python scripts/train.py --config configs/train_rgb_kpt.yaml
python scripts/evaluate.py --config configs/train_rgb_kpt.yaml --checkpoint path/to/model.ckpt
```

Training will report action classification metrics only:

- loss
- accuracy
- macro F1
- per-class accuracy
- optional confusion matrix export during evaluation

K-fold training will be supported through the engine layer, not embedded directly inside model or dataloader internals.

## Migration Plan

The legacy project package will be treated as source material. Code will be moved or rewritten into mvda with clearer module boundaries. Legacy files that are no longer used should be archived or removed after their useful logic is ported.

Priority order:

1. Create project metadata, package skeleton, configs, scripts, and README.
2. Build the unified data schema and datamodule with modality switches.
3. Port RGB and keypoint baseline models behind the unified classifier API.
4. Implement Lightning training and evaluation for action classification.
5. Add focused tests for config loading, dataset batch structure, model forward passes, and trainer selection-free execution.
6. Clean up old docs and legacy code paths after the new path is verified.

## Testing Strategy

Tests will focus on behavior rather than legacy implementation details:

- Config files load and validate.
- RGB-only, KPT-only, and RGB+KPT dummy batches pass through the model.
- Datamodule collate logic produces the normalized batch schema.
- Fusion modules handle one view and multiple views.
- Training step returns a valid loss on synthetic data.
- Evaluation metric aggregation works on synthetic predictions.

## Risks

The biggest risk is breaking working legacy assumptions during the full refactor. To reduce that risk, the refactor should first establish synthetic-data tests and small forward-pass tests before porting the full dataset loader. The second risk is hidden path assumptions in the old dataset code; these should be moved into config and path utilities early.

## Approval State

User selected the full refactor option and clarified that the project only needs action classification, with a configurable switch for RGB, keypoint, or RGB+keypoint loading.
