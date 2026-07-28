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


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Multi-class focal loss with optional per-class alpha weights."""
    weight = class_weights
    if weight is not None and weight.device != logits.device:
        weight = weight.to(logits.device)
    ce = F.cross_entropy(logits, labels.long(), weight=weight, reduction="none")
    pt = torch.exp(-F.cross_entropy(logits, labels.long(), reduction="none"))
    return ((1.0 - pt) ** gamma * ce).mean()


def balanced_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Balanced Softmax: shift logits by the log class prior during training.

    Rare classes (up is 9% of training segments, right 20% against down's 44%)
    are otherwise never predicted -- every model measured so far collapses onto
    the two frequent classes. Inference uses the unshifted logits, which is the
    point: the correction lives in the loss, not in the deployed model.
    """
    if prior.device != logits.device:
        prior = prior.to(logits.device)
    adjusted = logits + torch.log(prior.clamp_min(1e-12))
    weight = class_weights
    if weight is not None and weight.device != logits.device:
        weight = weight.to(logits.device)
    return F.cross_entropy(adjusted, labels.long(), weight=weight)


def build_class_prior(hparams) -> torch.Tensor | None:
    """Training class frequencies in `label_mapping_Dict` order, or None."""
    loss_cfg = getattr(hparams, "loss", None)
    values = getattr(loss_cfg, "class_prior", None) if loss_cfg is not None else None
    if values is None:
        return None
    prior = torch.tensor([float(v) for v in values], dtype=torch.float32)
    return prior / prior.sum()


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    hparams=None,
) -> torch.Tensor:
    """Dispatch on loss.type: 'ce' (default), 'focal', or 'balanced_softmax'."""
    loss_cfg = getattr(hparams, "loss", None) if hparams is not None else None
    loss_type = str(getattr(loss_cfg, "type", "ce")) if loss_cfg is not None else "ce"
    if loss_type == "focal":
        gamma = float(getattr(loss_cfg, "focal_gamma", 2.0))
        return focal_loss(logits, labels, class_weights, gamma=gamma)
    if loss_type == "balanced_softmax":
        prior = build_class_prior(hparams)
        if prior is None:
            raise ValueError(
                "loss.type=balanced_softmax requires loss.class_prior "
                "(training frequencies in left/right/down/up order)."
            )
        return balanced_softmax_loss(logits, labels, prior, class_weights)
    return weighted_cross_entropy(logits, labels, class_weights)
