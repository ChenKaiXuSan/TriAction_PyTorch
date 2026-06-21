#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Shared loss helpers for class-imbalanced action training."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from project.map_config import label_mapping_Dict


def build_class_weights(hparams, device: torch.device | None = None) -> torch.Tensor | None:
    """Build class weights ordered by `label_mapping_Dict`.

    Expected config:
        loss.class_weights:
          left: 1.0
          right: 1.0
          down: 1.0
          up: 4.0
          front: 0.2

    Returns None when class weighting is disabled or incomplete.
    """
    loss_cfg = getattr(hparams, "loss", None)
    weights_cfg = getattr(loss_cfg, "class_weights", None)
    if weights_cfg is None:
        return None

    num_classes = int(getattr(hparams.model, "model_class_num"))
    weights = []
    for class_id in range(num_classes):
        label_name = label_mapping_Dict[class_id]
        if isinstance(weights_cfg, Mapping):
            value = weights_cfg.get(label_name)
        else:
            value = getattr(weights_cfg, label_name, None)
        if value is None:
            return None
        weights.append(float(value))

    return torch.tensor(weights, dtype=torch.float32, device=device)


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Cross entropy with optional class weights on the logits device."""
    weight = class_weights
    if weight is not None and weight.device != logits.device:
        weight = weight.to(logits.device)
    return F.cross_entropy(logits, labels.long(), weight=weight)
