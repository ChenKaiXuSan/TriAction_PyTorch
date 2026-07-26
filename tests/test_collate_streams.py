"""Collate must carry every stream the model consumes, with aligned batch dims."""

import torch
from omegaconf import OmegaConf

from project.dataloader.data_loader import DriverDataModule


def _datamodule():
    cfg = OmegaConf.create(
        {
            "data": {
                "num_workers": 0,
                "val_num_workers": 0,
                "test_num_workers": 0,
                "prefetch_factor": 1,
                "img_size": 224,
                "uniform_temporal_subsample_num": 8,
                "batch_unit": "segment",
                "batch_size": 2,
                "max_video_frames": 1000,
                "segment_grouped_shuffle": False,
                "segment_shuffle_seed": 0,
                "head_roi_stream": True,
            },
            "model": {"input_type": "rgb_kpt", "model_class_num": 4},
            "train": {"view": "multi", "view_name": ["front", "left", "right"]},
            "paths": {"start_mid_end_path": "/nonexistent.json"},
            "experiment": "test",
        }
    )
    return DriverDataModule(cfg, {"train": [], "val": []})


def _sample():
    views = ["front", "left", "right"]
    return {
        "video": {v: torch.randn(3, 8, 224, 224) for v in views},
        "head_video": {v: torch.randn(3, 8, 224, 224) for v in views},
        "sam3d_kpt": {v: torch.randn(8, 70, 3) for v in views},
        "label": torch.tensor(1),
        "label_info": "right",
        "meta": {"experiment": "test"},
    }


def test_collate_keeps_head_video_stream():
    batch = _datamodule()._collate_fn([_sample(), _sample()])
    assert batch["head_video"] is not None, "head_video must survive collation"
    for view in ["front", "left", "right"]:
        assert batch["head_video"][view].shape == (2, 3, 8, 224, 224)


def test_collate_batch_dims_align_across_streams():
    batch = _datamodule()._collate_fn([_sample(), _sample()])
    dims = (
        {t.shape[0] for t in batch["video"].values()}
        | {t.shape[0] for t in batch["head_video"].values()}
        | {t.shape[0] for t in batch["sam3d_kpt"].values()}
        | {batch["label"].shape[0]}
    )
    assert dims == {2}, f"streams disagree on batch size: {dims}"
