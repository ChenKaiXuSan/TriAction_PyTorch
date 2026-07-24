#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
MV-ViViT: multi-view fusion on top of a (frozen) pretrained ViViT encoder.

Design informed by the 2026-07 experiment matrix
(doc/experiment_report_no_early_stopping_2026-07-24.md):
- ViViT is by far the strongest single-view backbone, but only had late
  (logit-mean) fusion; mid-level cross-view attention gave the biggest fusion
  gain on 3dcnn. This module brings token-level cross-view attention to ViViT.
- View embeddings are kept (removing them hurt TS-CVA), aggregation is plain
  mean pooling (gated aggregation hurt), fusion heads default to 8 (8 > 4 > 2).

Per view, the frozen shared ViViT encoder yields token features; each view is
compressed to a CLS token plus one spatially-pooled token per tubelet. Tokens
from all views (with additive view embeddings) form one sequence processed by
a small trainable transformer whose attention spans views — the cross-view
fusion. Mean pooling + linear head produce logits. With the backbone frozen
only the fusion blocks and head train (~15M params), which suits the small
dataset (80 training videos).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from project.models.hf_video_backbone import _load_transformers

DEFAULT_VIVIT_MODEL = "google/vivit-b-16x2-kinetics400"


class MVViVit(nn.Module):
    """Cross-view fusion classifier over a shared (frozen) ViViT encoder.

    forward() consumes ``videos[view] -> (B, C, T, H, W)`` and returns logits.
    """

    def __init__(self, hparams, view_names: Optional[List[str]] = None) -> None:
        super().__init__()
        model_cfg = hparams.model
        self.num_classes = int(model_cfg.model_class_num)
        self.view_names = list(view_names or ["front", "left", "right"])
        self.freeze_backbone = bool(
            getattr(model_cfg, "mv_vivit_freeze_backbone", True)
        )

        self.backbone = self._build_backbone(model_cfg)
        config = self.backbone.config
        self.feature_dim = int(config.hidden_size)
        # spatial patches per tubelet; token count is 1 (CLS) + tubelets * patches
        tubelet = config.tubelet_size  # [t, h, w]
        self.spatial_patches = (config.image_size // tubelet[1]) * (
            config.image_size // tubelet[2]
        )

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.use_view_embedding = bool(
            getattr(model_cfg, "mv_vivit_use_view_embedding", True)
        )
        if self.use_view_embedding:
            self.view_embedding = nn.Parameter(
                torch.zeros(len(self.view_names), 1, self.feature_dim)
            )
            nn.init.trunc_normal_(self.view_embedding, std=0.02)
        else:
            self.view_embedding = None

        num_layers = int(getattr(model_cfg, "mv_vivit_fusion_layers", 2))
        num_heads = int(getattr(model_cfg, "mv_vivit_fusion_heads", 8))
        ff_dim = int(getattr(model_cfg, "mv_vivit_fusion_ff_dim", 2048))
        dropout = float(getattr(model_cfg, "mv_vivit_dropout", 0.1))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.cross_view_fusion = nn.TransformerEncoder(
            fusion_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(self.feature_dim)
        self.head = nn.Linear(self.feature_dim, self.num_classes)

    @staticmethod
    def _build_backbone(model_cfg):
        transformers = _load_transformers()
        model_name = str(
            getattr(model_cfg, "vivit_model_name", None) or DEFAULT_VIVIT_MODEL
        )
        if bool(getattr(model_cfg, "hf_video_pretrained", True)):
            kwargs = {"add_pooling_layer": False}
            revision = getattr(model_cfg, "vivit_model_revision", None)
            if revision:
                kwargs["revision"] = str(revision)
            return transformers.VivitModel.from_pretrained(model_name, **kwargs)

        # random-init small config for offline tests / debugging
        hidden_size = int(getattr(model_cfg, "hf_video_hidden_size", 128))
        config = transformers.VivitConfig(
            hidden_size=hidden_size,
            num_hidden_layers=int(getattr(model_cfg, "hf_video_layers", 2)),
            num_attention_heads=int(getattr(model_cfg, "hf_video_heads", 4)),
            intermediate_size=int(
                getattr(model_cfg, "hf_video_intermediate_size", hidden_size * 4)
            ),
            image_size=int(getattr(model_cfg, "hf_video_image_size", 16)),
            num_frames=int(getattr(model_cfg, "hf_video_num_frames", 16)),
            tubelet_size=[
                int(getattr(model_cfg, "hf_video_tubelet_size", 2)),
                int(getattr(model_cfg, "hf_video_patch_size", 8)),
                int(getattr(model_cfg, "hf_video_patch_size", 8)),
            ],
        )
        return transformers.VivitModel(config, add_pooling_layer=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode_view(self, video: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, 1 + tubelets, D): CLS + per-tubelet pooled tokens."""
        pixel_values = video.float().permute(0, 2, 1, 3, 4).contiguous()
        if self.freeze_backbone:
            with torch.no_grad():
                hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        else:
            hidden = self.backbone(pixel_values=pixel_values).last_hidden_state

        cls_token = hidden[:, :1]
        patch_tokens = hidden[:, 1:]
        batch, num_patch_tokens, dim = patch_tokens.shape
        if num_patch_tokens % self.spatial_patches != 0:
            raise ValueError(
                f"Token count {num_patch_tokens} is not divisible by spatial patches "
                f"{self.spatial_patches}; input size must match the ViViT config."
            )
        tubelets = num_patch_tokens // self.spatial_patches
        temporal_tokens = patch_tokens.reshape(
            batch, tubelets, self.spatial_patches, dim
        ).mean(dim=2)
        return torch.cat([cls_token, temporal_tokens], dim=1)

    def forward(self, videos: Dict[str, torch.Tensor]) -> torch.Tensor:
        view_tokens = []
        for idx, view in enumerate(self.view_names):
            video = videos.get(view)
            if video is None:
                raise ValueError(f"MVViVit requires videos['{view}'].")
            tokens = self._encode_view(video)
            if self.view_embedding is not None:
                tokens = tokens + self.view_embedding[idx]
            view_tokens.append(tokens)

        sequence = torch.cat(view_tokens, dim=1)
        fused = self.cross_view_fusion(sequence)
        pooled = fused.mean(dim=1)
        return self.head(self.norm(pooled))
