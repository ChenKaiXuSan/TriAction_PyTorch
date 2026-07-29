import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("transformers")

from project.models.mv_vivit import MVViVit


def _hparams(**model_overrides):
    model = {
        "model_class_num": 4,
        "hf_video_pretrained": False,
        "hf_video_hidden_size": 64,
        "hf_video_layers": 4,
        "hf_video_heads": 4,
        "hf_video_image_size": 16,
        "hf_video_num_frames": 8,
        "hf_video_tubelet_size": 2,
        "hf_video_patch_size": 8,
        "mv_vivit_fusion_layers": 1,
        "mv_vivit_fusion_heads": 4,
        "mv_vivit_fusion_ff_dim": 128,
    }
    model.update(model_overrides)
    return OmegaConf.create({"model": model})


def _videos(batch=2):
    return {v: torch.randn(batch, 3, 8, 16, 16) for v in ["front", "left", "right"]}


def test_frozen_backbone_forward_and_grads():
    model = MVViVit(_hparams())
    logits = model(_videos())
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert all(
        p.grad is None for n, p in model.named_parameters() if n.startswith("backbone.")
    )
    assert all(
        p.grad is not None for n, p in model.named_parameters() if p.requires_grad
    )


def test_partial_unfreeze_grads_reach_only_last_layers():
    model = MVViVit(_hparams(mv_vivit_unfreeze_last_layers=2))
    logits = model(_videos())
    logits.sum().backward()
    layers = model._encoder_layers(model.backbone)
    assert any(p.grad is not None for p in layers[-1].parameters())
    assert any(p.grad is not None for p in layers[-2].parameters())
    assert all(p.grad is None for p in layers[0].parameters())
    assert all(p.grad is None for p in model.backbone.embeddings.parameters())


def test_view_logit_ensemble_shape():
    model = MVViVit(_hparams(mv_vivit_view_logit_ensemble=True))
    assert model.view_head is not None
    logits = model(_videos())
    assert logits.shape == (2, 4)


def test_kpt_stream_token_forward():
    model = MVViVit(_hparams(mv_vivit_kpt_stream=True))
    kpts = {v: torch.randn(2, 8, 70, 3) for v in ["front", "left", "right"]}
    logits = model(_videos(), kpts=kpts)
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert any(p.grad is not None for p in model.kpt_encoder.parameters())


def test_head_stream_forward():
    model = MVViVit(_hparams(mv_vivit_head_stream=True))
    logits = model(_videos(), head_videos=_videos())
    assert logits.shape == (2, 4)


def test_kpt_query_pooling_forward():
    model = MVViVit(_hparams(mv_vivit_kpt_query_pooling=True))
    assert model.query_pool is not None
    kpts = {v: torch.randn(2, 8, 70, 3) for v in ["front", "left", "right"]}
    logits = model(_videos(), kpts=kpts)
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert any(p.grad is not None for p in model.query_pool.parameters())


def test_aux_pose_head_and_targets():
    from project.trainer.multi.mv.train_mv_vivit import head_pose_targets

    model = MVViVit(_hparams(mv_vivit_aux_pose_weight=0.5))
    assert model.aux_pose_head is not None
    logits, pooled = model(_videos(), return_pooled=True)
    aux = model.aux_pose_head(pooled)
    assert aux.shape == (2, 2 * model.aux_pose_steps)
    target = head_pose_targets(torch.randn(2, 12, 70, 3), model.aux_pose_steps)
    assert target.shape == aux.shape


def test_head_pose_stream_forward_and_grads():
    model = MVViVit(_hparams(mv_vivit_head_pose_stream=True))
    assert model.head_pose_proj is not None
    kpts = {v: torch.randn(2, 8, 70, 3) for v in ["front", "left", "right"]}
    logits = model(_videos(), kpts=kpts)
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert any(p.grad is not None for p in model.head_pose_proj.parameters())
