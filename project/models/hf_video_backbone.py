#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Hugging Face video-classification backbones for RGB action recognition."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class _HFVideoSpec:
    config_cls: str
    model_cls: str
    default_model_name: str
    model_name_attr: str


_SPECS = {
    "videomae": _HFVideoSpec(
        config_cls="VideoMAEConfig",
        model_cls="VideoMAEForVideoClassification",
        default_model_name="MCG-NJU/videomae-base-finetuned-kinetics",
        model_name_attr="videomae_model_name",
    ),
    "vivit": _HFVideoSpec(
        config_cls="VivitConfig",
        model_cls="VivitForVideoClassification",
        default_model_name="google/vivit-b-16x2-kinetics400",
        model_name_attr="vivit_model_name",
    ),
}


def _load_transformers():
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - exercised in environments without dependency
        raise ImportError(
            "Hugging Face video backbones require the optional 'transformers' package. "
            "Install it with `pip install transformers accelerate safetensors`."
        ) from exc


class HFVideoClassificationBackbone(nn.Module):
    """Wrap HF video classifiers while keeping the local ``(B, C, T, H, W)`` API."""

    def __init__(self, hparams, family: str) -> None:
        super().__init__()
        if family not in _SPECS:
            raise ValueError(f"Unsupported Hugging Face video backbone: {family}")

        self.family = family
        self.model_cfg = hparams.model
        self.num_classes = int(self.model_cfg.model_class_num)
        self.spec = _SPECS[family]
        model_name = (
            getattr(self.model_cfg, "hf_video_model_name", None)
            or getattr(self.model_cfg, self.spec.model_name_attr, None)
            or self.spec.default_model_name
        )
        self.model_name = str(model_name)
        model_revision = (
            getattr(self.model_cfg, "hf_video_revision", None)
            or getattr(self.model_cfg, f"{family}_model_revision", None)
        )

        transformers = _load_transformers()
        config_cls = getattr(transformers, self.spec.config_cls)
        model_cls = getattr(transformers, self.spec.model_cls)

        use_pretrained = bool(getattr(self.model_cfg, "hf_video_pretrained", True))
        if use_pretrained:
            pretrained_kwargs = {
                "num_labels": self.num_classes,
                "ignore_mismatched_sizes": bool(
                    getattr(self.model_cfg, "hf_video_ignore_mismatched_sizes", True)
                ),
            }
            if model_revision:
                pretrained_kwargs["revision"] = str(model_revision)
            self.model = model_cls.from_pretrained(
                self.model_name,
                **pretrained_kwargs,
            )
        else:
            hidden_size = int(getattr(self.model_cfg, "hf_video_hidden_size", 128))
            config = self._build_random_config(config_cls, hidden_size)
            self.model = model_cls(config)

        self.feature_dim = int(getattr(self.model.config, "hidden_size", 0))
        if self.feature_dim <= 0:
            raise ValueError(
                f"{type(self.model).__name__} config must expose a positive hidden_size."
            )

    def _build_random_config(self, config_cls, hidden_size: int):
        common_kwargs = {
            "num_labels": self.num_classes,
            "hidden_size": hidden_size,
        }
        if self.family == "videomae":
            common_kwargs.update(
                {
                    "num_hidden_layers": int(getattr(self.model_cfg, "hf_video_layers", 2)),
                    "num_attention_heads": int(getattr(self.model_cfg, "hf_video_heads", 4)),
                    "intermediate_size": int(
                        getattr(self.model_cfg, "hf_video_intermediate_size", hidden_size * 4)
                    ),
                    "image_size": int(getattr(self.model_cfg, "hf_video_image_size", 16)),
                    "num_frames": int(getattr(self.model_cfg, "hf_video_num_frames", 16)),
                    "tubelet_size": int(getattr(self.model_cfg, "hf_video_tubelet_size", 2)),
                    "patch_size": int(getattr(self.model_cfg, "hf_video_patch_size", 8)),
                }
            )
        else:
            common_kwargs.update(
                {
                    "num_hidden_layers": int(getattr(self.model_cfg, "hf_video_layers", 2)),
                    "num_attention_heads": int(getattr(self.model_cfg, "hf_video_heads", 4)),
                    "intermediate_size": int(
                        getattr(self.model_cfg, "hf_video_intermediate_size", hidden_size * 4)
                    ),
                    "image_size": int(getattr(self.model_cfg, "hf_video_image_size", 16)),
                    "num_frames": int(getattr(self.model_cfg, "hf_video_num_frames", 16)),
                    "tubelet_size": [
                        int(getattr(self.model_cfg, "hf_video_tubelet_size", 2)),
                        int(getattr(self.model_cfg, "hf_video_patch_size", 8)),
                        int(getattr(self.model_cfg, "hf_video_patch_size", 8)),
                    ],
                }
            )
        return config_cls(**common_kwargs)

    @staticmethod
    def _prepare_video(video: torch.Tensor) -> torch.Tensor:
        if video is None:
            raise ValueError("RGB video input is required.")
        if video.ndim != 5:
            raise ValueError(f"Expected RGB video with shape (B, C, T, H, W), got {tuple(video.shape)}")
        return video.float().permute(0, 2, 1, 3, 4).contiguous()

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_video(video)
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError(
                f"{type(self.model).__name__} did not return hidden states for feature fusion."
            )
        last_hidden = hidden_states[-1]
        if last_hidden.ndim == 3:
            return last_hidden.mean(dim=1)
        if last_hidden.ndim == 2:
            return last_hidden
        raise ValueError(f"Unexpected hidden state shape: {tuple(last_hidden.shape)}")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_video(video)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


class VideoMAEBackbone(HFVideoClassificationBackbone):
    def __init__(self, hparams) -> None:
        super().__init__(hparams, family="videomae")


class VivitBackbone(HFVideoClassificationBackbone):
    def __init__(self, hparams) -> None:
        super().__init__(hparams, family="vivit")
