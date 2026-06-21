# Modality And View Selectable Training Design

## Goal

Support classification experiments with selectable input modality and selectable camera views:

- `model.input_type=rgb`
- `model.input_type=kpt`
- `model.input_type=rgb_kpt`
- `train.view_name` as any non-empty subset of `front`, `left`, and `right`

## Scope

This change focuses on the shared data contract and classification trainers used by the current training entrypoint. It does not redesign TS-CVA internals or early-fusion legacy behavior.

## Data Contract

`DriverDataModule` derives modality loading from `model.input_type`.

- `rgb`: load only RGB video tensors.
- `kpt`: load only SAM3D keypoint tensors.
- `rgb_kpt`: load both RGB and SAM3D keypoint tensors.

Each batch keeps stable optional keys:

```python
batch["video"]      # None or Dict[str, Tensor[B, C, T, H, W]]
batch["sam3d_kpt"]  # None or Dict[str, Tensor[B, T, K, 3]]
batch["label"]      # Tensor[B]
```

Only views listed in `train.view_name` are loaded and collated.

## Model And Trainer Routing

Trainer selection accepts all three input types. A shared classification trainer handles:

- one or more selected views,
- RGB-only classification,
- KPT-only classification,
- RGB+KPT classification,
- view-level fusion by averaging logits across selected views.

RGB uses the existing video backbones. KPT uses a lightweight temporal keypoint classifier. RGB+KPT combines per-view RGB and KPT logits with an average, keeping the first implementation simple and easy to compare.

## Testing

Tests cover:

- dataloader load flags inferred from `input_type`,
- collate behavior for optional RGB/KPT data and selected views,
- trainer selection for `rgb`, `kpt`, and `rgb_kpt`,
- forward/training-step behavior for selectable views.
