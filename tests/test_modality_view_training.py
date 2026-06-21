import torch
from omegaconf import OmegaConf

from project.dataloader.data_loader import DriverDataModule
from project.models.keypoint_classifier import KeypointTemporalClassifier
from project.trainer.single_selector import select_single_trainer_cls


def _base_cfg(input_type="rgb", view_name=None):
    return OmegaConf.create(
        {
            "experiment": "test",
            "log_path": "/tmp/modality-view-test",
            "paths": {"start_mid_end_path": "/tmp/anno.json"},
            "data": {
                "num_workers": 0,
                "img_size": 16,
                "uniform_temporal_subsample_num": 4,
                "batch_size": 1,
                "max_video_frames": 100,
            },
            "loss": {"lr": 0.001},
            "model": {
                "input_type": input_type,
                "backbone": "3dcnn",
                "model_class_num": 4,
                "kpt_hidden_dim": 8,
                "kpt_dropout": 0.0,
            },
            "train": {
                "view": "single",
                "view_name": view_name or ["front"],
            },
        }
    )


def test_data_module_derives_load_flags_from_input_type():
    rgb_dm = DriverDataModule(_base_cfg("rgb"), {"train": [], "val": []})
    assert rgb_dm.load_rgb is True
    assert rgb_dm.load_kpt is False

    kpt_dm = DriverDataModule(_base_cfg("kpt"), {"train": [], "val": []})
    assert kpt_dm.load_rgb is False
    assert kpt_dm.load_kpt is True

    fused_dm = DriverDataModule(_base_cfg("rgb_kpt"), {"train": [], "val": []})
    assert fused_dm.load_rgb is True
    assert fused_dm.load_kpt is True


def test_collate_keeps_only_requested_modalities_and_views():
    dm = DriverDataModule(
        _base_cfg("kpt", view_name=["front", "left"]),
        {"train": [], "val": []},
    )

    sample = {
        "video": None,
        "sam3d_kpt": {
            "front": torch.ones(2, 4, 6, 3),
            "left": torch.ones(2, 4, 6, 3) * 2,
        },
        "label": torch.tensor([0, 1]),
        "label_info": ["left", "right"],
        "meta": {"person_id": "01", "chunk_info": None},
    }

    batch = dm._collate_fn([sample])

    assert batch["video"] is None
    assert set(batch["sam3d_kpt"]) == {"front", "left"}
    assert batch["sam3d_kpt"]["front"].shape == (2, 4, 6, 3)
    assert batch["sam3d_kpt"]["left"].shape == (2, 4, 6, 3)
    assert batch["label"].tolist() == [0, 1]


def test_single_selector_accepts_all_classification_input_types():
    rgb_cls = select_single_trainer_cls(_base_cfg("rgb"))
    kpt_cls = select_single_trainer_cls(_base_cfg("kpt"))
    fused_cls = select_single_trainer_cls(_base_cfg("rgb_kpt"))

    assert rgb_cls is kpt_cls is fused_cls


def test_keypoint_classifier_forward_shape():
    model = KeypointTemporalClassifier(_base_cfg("kpt"))
    logits = model(torch.randn(3, 5, 6, 3))
    assert logits.shape == (3, 4)
