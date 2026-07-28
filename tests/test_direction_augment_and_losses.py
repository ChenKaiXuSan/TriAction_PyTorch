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


def test_clip_mean_subtraction_removes_static_content_keeps_motion():
    # a static background plus a moving component; after subtracting the clip
    # mean only the motion should survive
    static = torch.randn(3, 1, 8, 8).repeat(1, 6, 1, 1)
    motion = torch.zeros(3, 6, 8, 8)
    motion[:, 3:] = 1.0
    clip = static + motion

    class _DS:
        subtract_clip_mean = True
        _maybe_subtract_clip_mean = LabeledVideoDataset._maybe_subtract_clip_mean

    out = _DS()._maybe_subtract_clip_mean(clip)
    assert torch.allclose(out.mean(dim=1), torch.zeros(3, 8, 8), atol=1e-5)
    # frames before/after the step must remain distinguishable
    assert (out[:, 0] - out[:, 5]).abs().mean() > 0.5


def test_clip_mean_subtraction_is_opt_in():
    clip = torch.randn(3, 4, 8, 8)

    class _DS:
        subtract_clip_mean = False
        _maybe_subtract_clip_mean = LabeledVideoDataset._maybe_subtract_clip_mean

    assert torch.equal(_DS()._maybe_subtract_clip_mean(clip), clip)


def test_balanced_softmax_favours_rare_classes():
    from project.trainer.losses import balanced_softmax_loss

    prior = torch.tensor([0.26, 0.20, 0.44, 0.09])
    logits = torch.zeros(2, 4)
    rare = torch.tensor([3, 3])       # "up", the rarest class
    frequent = torch.tensor([2, 2])   # "down", the most frequent
    # shifting by log-prior makes a rare-class mistake cost more than a
    # frequent-class one, which is what pushes the model off the majority class
    assert balanced_softmax_loss(logits, rare, prior) > balanced_softmax_loss(
        logits, frequent, prior
    )


def test_classification_loss_requires_prior_for_balanced_softmax():
    import pytest
    from project.trainer.losses import classification_loss

    hparams = OmegaConf.create({"loss": {"type": "balanced_softmax"}})
    with pytest.raises(ValueError, match="class_prior"):
        classification_loss(torch.randn(4, 4), torch.randint(0, 4, (4,)), None, hparams)
