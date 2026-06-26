import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from project.trainer.multi.early.train_early_fusion import EarlyFusion3DCNNTrainer


class DummyVideoModel(torch.nn.Module):
    feature_dim = 1

    def __init__(self, hparams):
        super().__init__()
        self.num_classes = int(hparams.model.model_class_num)
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward_features(self, video):
        return video.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1) * self.scale

    def forward(self, video):
        return self.forward_features(video).repeat(1, self.num_classes)


def _cfg(fuse_method="avg"):
    return OmegaConf.create(
        {
            "data": {"img_size": 16},
            "loss": {"lr": 0.001},
            "model": {
                "input_type": "rgb",
                "backbone": "3dcnn",
                "model_class_num": 3,
                "fuse_method": fuse_method,
            },
            "train": {"view_name": ["front", "left", "right"]},
        }
    )


def _batch():
    return {
        "video": {
            "front": torch.ones(2, 3, 4, 5, 5),
            "left": torch.ones(2, 3, 4, 5, 5) * 2,
            "right": torch.ones(2, 3, 4, 5, 5) * 3,
        },
        "label": torch.tensor([0, 1]),
    }


class EarlyFusionTrainerTest(unittest.TestCase):
    def test_validation_step_accepts_multiview_video_dict(self):
        with patch(
            "project.trainer.multi.early.train_early_fusion.select_model",
            DummyVideoModel,
        ):
            trainer = EarlyFusion3DCNNTrainer(_cfg("avg"))
            trainer.validation_step(_batch(), 0)

    def test_concat_forward_accepts_multiview_video_dict(self):
        with patch(
            "project.trainer.multi.early.train_early_fusion.select_model",
            DummyVideoModel,
        ):
            trainer = EarlyFusion3DCNNTrainer(_cfg("concat"))
            logits = trainer(_batch()["video"])

        self.assertEqual(tuple(logits.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
