from omegaconf import OmegaConf
import torch

from project.trainer.single_selector import select_single_trainer_cls
from project.trainer.multi_selector import select_multi_trainer_cls
from project.trainer.single.train_single_modality import SingleModalityClassifierTrainer
import project.trainer.multi.early.train_early_fusion as early_fusion
from project.trainer.multi.early.train_early_fusion import EarlyFusion3DCNNTrainer
from project.trainer.multi.late.train_late_fusion import LateFusion3DCNNTrainer
from project.trainer.multi.mid.train_multi_ts_cva import MultiTSCVATrainer


def test_select_single_trainer_rgb_3dcnn():
    hparams = OmegaConf.create(
        {"train": {"view": "single"}, "model": {"input_type": "rgb", "backbone": "3dcnn"}}
    )
    trainer_cls = select_single_trainer_cls(hparams)
    assert trainer_cls is SingleModalityClassifierTrainer


def test_select_multi_trainer_early_fusion():
    hparams = OmegaConf.create(
        {
            "train": {"view": "multi"},
            "model": {
                "input_type": "rgb",
                "backbone": "3dcnn",
                "fuse_method": "avg",
            },
        }
    )
    trainer_cls = select_multi_trainer_cls(hparams)
    assert trainer_cls is EarlyFusion3DCNNTrainer


def test_select_multi_trainer_mid_fusion():
    hparams = OmegaConf.create(
        {
            "train": {"view": "multi"},
            "model": {
                "input_type": "rgb",
                "backbone": "3dcnn",
                "fuse_method": "mid",
            },
        }
    )
    trainer_cls = select_multi_trainer_cls(hparams)
    assert trainer_cls is MultiTSCVATrainer


def test_select_multi_trainer_late_fusion():
    hparams = OmegaConf.create(
        {
            "train": {"view": "multi"},
            "model": {
                "input_type": "rgb",
                "backbone": "3dcnn",
                "fuse_method": "late",
            },
        }
    )
    trainer_cls = select_multi_trainer_cls(hparams)
    assert trainer_cls is LateFusion3DCNNTrainer


def test_early_fusion_trainer_uses_loss_lr_without_optimizer(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self, hparams):
            super().__init__()
            self.linear = torch.nn.Linear(1, hparams.model.model_class_num)

        def forward(self, x):
            return self.linear(torch.zeros(x.shape[0], 1, device=x.device))

    monkeypatch.setattr(early_fusion, "select_model", DummyModel)
    hparams = OmegaConf.create(
        {
            "data": {"img_size": 224},
            "loss": {"lr": 0.0001},
            "model": {
                "model_class_num": 5,
                "input_type": "rgb",
                "backbone": "3dcnn",
            },
        }
    )

    trainer = EarlyFusion3DCNNTrainer(hparams)

    assert trainer.lr == 0.0001
