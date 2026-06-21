#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Lightweight classifiers for SAM3D keypoint and RGB+KPT experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


class KeypointTemporalClassifier(nn.Module):
    """Temporal classifier for keypoints shaped as ``(B, T, K, 3)``."""

    def __init__(self, hparams) -> None:
        super().__init__()
        model_cfg = hparams.model
        self.num_classes = int(model_cfg.model_class_num)
        hidden_dim = int(getattr(model_cfg, "kpt_hidden_dim", 128))
        dropout = float(getattr(model_cfg, "kpt_dropout", 0.1))

        self.input_proj = nn.LazyLinear(hidden_dim)
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.feature_dim = hidden_dim

    def forward_features(self, kpts: torch.Tensor) -> torch.Tensor:
        if kpts is None:
            raise ValueError("Keypoint input is required.")
        if kpts.ndim != 4 or kpts.shape[-1] != 3:
            raise ValueError(
                f"Expected keypoints with shape (B, T, K, 3), got {tuple(kpts.shape)}"
            )

        x = kpts.float().flatten(start_dim=2)
        x = self.input_proj(x)
        x, _ = self.temporal_encoder(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, kpts: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(kpts))


class RGBKPTClassifier(nn.Module):
    """Per-view RGB+KPT classifier that averages modality logits."""

    def __init__(self, rgb_model: nn.Module, kpt_model: nn.Module) -> None:
        super().__init__()
        self.rgb_model = rgb_model
        self.kpt_model = kpt_model

    def forward(self, video: torch.Tensor, kpts: torch.Tensor) -> torch.Tensor:
        if video is None or kpts is None:
            raise ValueError("RGB+KPT input requires both video and keypoints.")
        return (self.rgb_model(video) + self.kpt_model(kpts)) / 2
