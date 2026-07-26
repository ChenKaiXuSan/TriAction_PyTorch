import torch
from omegaconf import OmegaConf

from project.dataloader.whole_video_dataset import LabeledVideoDataset
from project.trainer.losses import classification_loss, focal_loss


def _videos(views):
    return {v: torch.arange(24.0).reshape(1, 2, 3, 4) + i for i, v in enumerate(views)}


def test_mirror_aug_three_views_swaps_cameras_and_label():
    videos = _videos(["front", "left", "right"])
    flipped, label = LabeledVideoDataset._apply_mirror_direction_aug(videos, "left")
    assert label == "right"
    assert torch.equal(flipped["left"], torch.flip(videos["right"], dims=[-1]))
    assert torch.equal(flipped["right"], torch.flip(videos["left"], dims=[-1]))
    assert torch.equal(flipped["front"], torch.flip(videos["front"], dims=[-1]))


def test_mirror_aug_front_only_flips_and_swaps_label():
    videos = _videos(["front"])
    flipped, label = LabeledVideoDataset._apply_mirror_direction_aug(videos, "right")
    assert label == "left"
    assert torch.equal(flipped["front"], torch.flip(videos["front"], dims=[-1]))


def test_mirror_aug_vertical_labels_unchanged():
    videos = _videos(["front", "left", "right"])
    _, label = LabeledVideoDataset._apply_mirror_direction_aug(videos, "up")
    assert label == "up"


def test_mirror_aug_unsupported_view_set_is_noop():
    videos = _videos(["left"])
    flipped, label = LabeledVideoDataset._apply_mirror_direction_aug(videos, "left")
    assert label == "left"
    assert torch.equal(flipped["left"], videos["left"])


def test_focal_loss_matches_ce_at_gamma_zero():
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 4, (8,))
    ce = torch.nn.functional.cross_entropy(logits, labels)
    fl = focal_loss(logits, labels, class_weights=None, gamma=0.0)
    assert torch.allclose(ce, fl, atol=1e-6)


def test_classification_loss_dispatches_focal():
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 4, (8,))
    hparams = OmegaConf.create({"loss": {"type": "focal", "focal_gamma": 2.0}})
    fl = classification_loss(logits, labels, None, hparams=hparams)
    ce = classification_loss(logits, labels, None, hparams=None)
    assert fl.item() <= ce.item() + 1e-6
