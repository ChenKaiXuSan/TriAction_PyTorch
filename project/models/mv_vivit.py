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
from project.models.head_pose_features import FEATURE_DIM as HEAD_POSE_DIM
from project.models.head_pose_features import head_pose_features

DEFAULT_VIVIT_MODEL = "google/vivit-b-16x2-kinetics400"


class _TrajectoryEncoder(nn.Module):
    """(B, T, K, 3) keypoint trajectory -> one (B, D) token via GRU.

    The input layer is lazy because the dataset yields the raw SAM-3D
    keypoint set, whose size we don't hard-code here.
    """

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, kpts: torch.Tensor) -> torch.Tensor:
        if kpts.ndim != 4:
            raise ValueError(f"Expected keypoints (B, T, K, 3), got {tuple(kpts.shape)}")
        batch, steps = kpts.shape[0], kpts.shape[1]
        x = self.proj(kpts.reshape(batch, steps, -1).float())
        _, hidden = self.gru(x)
        return self.out(hidden[-1])


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

        # partial finetuning: keep the last N encoder layers (+ final layernorm)
        # trainable while the rest of the backbone stays frozen
        self.unfreeze_last = int(
            getattr(model_cfg, "mv_vivit_unfreeze_last_layers", 0)
        )
        if self.unfreeze_last < 0:
            raise ValueError("mv_vivit_unfreeze_last_layers must be >= 0")
        self.unfreeze_last = min(
            self.unfreeze_last, len(self._encoder_layers(self.backbone))
        )

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            if self.unfreeze_last > 0:
                for layer in self._encoder_layers(self.backbone)[-self.unfreeze_last :]:
                    layer.requires_grad_(True)
                self.backbone.layernorm.requires_grad_(True)
            else:
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

        # optional hybrid: average the fusion logits with per-view logits from a
        # shared auxiliary head (late-fusion-style ensemble inside the model)
        self.view_logit_ensemble = bool(
            getattr(model_cfg, "mv_vivit_view_logit_ensemble", False)
        )
        if self.view_logit_ensemble:
            self.view_head = nn.Linear(self.feature_dim, self.num_classes)
        else:
            self.view_head = None

        # optional per-view keypoint-trajectory token (head pose is nearly a
        # direct readout of head-movement direction); appended to the fusion
        # sequence so cross-view attention can consume it
        self.use_kpt_stream = bool(getattr(model_cfg, "mv_vivit_kpt_stream", False))
        # kpt-query pooling replaces mean pooling with cross-attention whose
        # queries are the per-view keypoint-trajectory tokens (level-3 guidance)
        self.use_kpt_query_pooling = bool(
            getattr(model_cfg, "mv_vivit_kpt_query_pooling", False)
        )
        if self.use_kpt_stream or self.use_kpt_query_pooling:
            kpt_hidden = int(getattr(model_cfg, "mv_vivit_kpt_hidden", 256))
            self.kpt_encoder = _TrajectoryEncoder(
                hidden_dim=kpt_hidden, out_dim=self.feature_dim
            )
        else:
            self.kpt_encoder = None
        if self.use_kpt_query_pooling:
            num_heads = int(getattr(model_cfg, "mv_vivit_fusion_heads", 8))
            self.query_pool = nn.MultiheadAttention(
                self.feature_dim, num_heads, batch_first=True
            )
        else:
            self.query_pool = None

        # head-ROI dual stream: head crops share the backbone; their tokens get
        # an additive stream embedding so fusion can tell the streams apart
        self.use_head_stream = bool(getattr(model_cfg, "mv_vivit_head_stream", False))
        if self.use_head_stream:
            self.head_stream_embedding = nn.Parameter(
                torch.zeros(1, 1, self.feature_dim)
            )
            nn.init.trunc_normal_(self.head_stream_embedding, std=0.02)
        else:
            self.head_stream_embedding = None

        # head-pose stream v2: analytic per-frame pose features (angles,
        # deltas, velocity, local shape) projected to one token per frame per
        # view. Unlike the raw-kpt streams this feeds the head signal directly:
        # the raw 70-point tensor is 60% hand rows and its absolute coordinates
        # encode driver identity, which is why the earlier kpt arms measured
        # no contribution.
        self.use_head_pose_stream = bool(
            getattr(model_cfg, "mv_vivit_head_pose_stream", False)
        )
        if self.use_head_pose_stream:
            self.head_pose_proj = nn.Sequential(
                nn.Linear(HEAD_POSE_DIM, self.feature_dim), nn.GELU()
            )
            self.head_pose_stream_embedding = nn.Parameter(
                torch.zeros(1, 1, self.feature_dim)
            )
            nn.init.trunc_normal_(self.head_pose_stream_embedding, std=0.02)
        else:
            self.head_pose_proj = None
            self.head_pose_stream_embedding = None

        # CONTROL for the head-pose stream: identical per-frame token pipeline
        # but fed the raw flattened keypoints instead of the analytic features.
        # Separates "the analytic representation matters" from "any extra
        # per-frame tokens help".
        self.use_raw_pose_stream = bool(
            getattr(model_cfg, "mv_vivit_raw_pose_stream", False)
        )
        if self.use_raw_pose_stream:
            self.raw_pose_proj = nn.Sequential(
                nn.LazyLinear(self.feature_dim), nn.GELU()
            )
            self.raw_pose_stream_embedding = nn.Parameter(
                torch.zeros(1, 1, self.feature_dim)
            )
            nn.init.trunc_normal_(self.raw_pose_stream_embedding, std=0.02)
        else:
            self.raw_pose_proj = None
            self.raw_pose_stream_embedding = None

        # auxiliary head-pose regression (level-4 guidance): trained from
        # keypoint-derived yaw/pitch targets, unused at inference
        self.aux_pose_steps = int(getattr(model_cfg, "mv_vivit_aux_pose_steps", 8))
        aux_weight = float(getattr(model_cfg, "mv_vivit_aux_pose_weight", 0.0))
        if aux_weight > 0:
            self.aux_pose_head = nn.Linear(self.feature_dim, 2 * self.aux_pose_steps)
        else:
            self.aux_pose_head = None

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
            if self.unfreeze_last > 0:
                for layer in self._encoder_layers(self.backbone)[-self.unfreeze_last :]:
                    layer.train(mode)
        return self

    @staticmethod
    def _encoder_layers(backbone):
        # transformers >= 5 exposes VivitModel.layers; older versions encoder.layer
        if hasattr(backbone, "layers"):
            return backbone.layers
        return backbone.encoder.layer

    @staticmethod
    def _layer_output(layer_out):
        # VivitLayer.forward returns a Tensor in recent transformers, a tuple in older ones
        return layer_out[0] if isinstance(layer_out, tuple) else layer_out

    def _backbone_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        fully_frozen = self.freeze_backbone and self.unfreeze_last == 0
        if fully_frozen:
            with torch.no_grad():
                return self.backbone(pixel_values=pixel_values).last_hidden_state
        if not self.freeze_backbone:
            return self.backbone(pixel_values=pixel_values).last_hidden_state

        # partial finetune: frozen prefix without autograd, trainable suffix with it
        layers = self._encoder_layers(self.backbone)
        split = len(layers) - self.unfreeze_last
        with torch.no_grad():
            hidden = self.backbone.embeddings(pixel_values)
            for layer in layers[:split]:
                hidden = self._layer_output(layer(hidden))
        for layer in layers[split:]:
            hidden = self._layer_output(layer(hidden))
        return self.backbone.layernorm(hidden)

    def _encode_view(self, video: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, 1 + tubelets, D): CLS + per-tubelet pooled tokens."""
        pixel_values = video.float().permute(0, 2, 1, 3, 4).contiguous()
        hidden = self._backbone_tokens(pixel_values)

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

    def _view_kpt_token(
        self, kpts: Optional[Dict[str, torch.Tensor]], view: str
    ) -> torch.Tensor:
        if kpts is None or kpts.get(view) is None:
            raise ValueError(
                f"keypoint guidance is enabled but keypoints for view '{view}' are missing."
            )
        return self.kpt_encoder(kpts[view])

    def forward(
        self,
        videos: Dict[str, torch.Tensor],
        kpts: Optional[Dict[str, torch.Tensor]] = None,
        head_videos: Optional[Dict[str, torch.Tensor]] = None,
        return_pooled: bool = False,
    ):
        view_tokens = []
        kpt_query_tokens = []
        for idx, view in enumerate(self.view_names):
            video = videos.get(view)
            if video is None:
                raise ValueError(f"MVViVit requires videos['{view}'].")
            tokens = self._encode_view(video)
            if self.use_head_stream:
                if head_videos is None or head_videos.get(view) is None:
                    raise ValueError(
                        f"mv_vivit_head_stream=true requires head_videos['{view}'] "
                        "(enable data.head_roi_stream)."
                    )
                head_tokens = (
                    self._encode_view(head_videos[view]) + self.head_stream_embedding
                )
                tokens = torch.cat([tokens, head_tokens], dim=1)
            if self.head_pose_proj is not None:
                if kpts is None or kpts.get(view) is None:
                    raise ValueError(
                        f"mv_vivit_head_pose_stream=true requires keypoints for view "
                        f"'{view}' (set model.input_type=rgb_kpt)."
                    )
                pose_feats = head_pose_features(kpts[view])
                pose_tokens = (
                    self.head_pose_proj(pose_feats) + self.head_pose_stream_embedding
                )
                tokens = torch.cat([tokens, pose_tokens], dim=1)
            if self.raw_pose_proj is not None:
                if kpts is None or kpts.get(view) is None:
                    raise ValueError(
                        f"mv_vivit_raw_pose_stream=true requires keypoints for view "
                        f"'{view}' (set model.input_type=rgb_kpt)."
                    )
                raw = kpts[view].float().flatten(2)  # (B, T, K*3)
                raw_tokens = (
                    self.raw_pose_proj(raw) + self.raw_pose_stream_embedding
                )
                tokens = torch.cat([tokens, raw_tokens], dim=1)
            if self.kpt_encoder is not None:
                kpt_token = self._view_kpt_token(kpts, view)
                kpt_query_tokens.append(kpt_token)
                if self.use_kpt_stream:
                    tokens = torch.cat([tokens, kpt_token.unsqueeze(1)], dim=1)
            if self.view_embedding is not None:
                tokens = tokens + self.view_embedding[idx]
            view_tokens.append(tokens)

        sequence = torch.cat(view_tokens, dim=1)
        fused = self.cross_view_fusion(sequence)

        if self.query_pool is not None:
            queries = torch.stack(kpt_query_tokens, dim=1)  # (B, V, D)
            attended, _ = self.query_pool(queries, fused, fused)
            pooled = attended.mean(dim=1)
        else:
            pooled = fused.mean(dim=1)
        logits = self.head(self.norm(pooled))

        if self.view_head is not None:
            view_logits = torch.stack(
                [self.view_head(self.norm(t.mean(dim=1))) for t in view_tokens]
            ).mean(dim=0)
            logits = 0.5 * logits + 0.5 * view_logits
        if return_pooled:
            return logits, self.norm(pooled)
        return logits
