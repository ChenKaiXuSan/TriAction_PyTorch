#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Shared classification metrics for the trainers.

Metrics must be *accumulated* across an epoch and reduced once, not computed
per batch and averaged. With small batches (segment batching runs at
batch_size 2) the per-batch average is badly biased: measured on the
2026-07-26 runs it reported macro F1 0.52 where the true epoch value was 0.35.

Usage in a LightningModule::

    self.train_metrics, self.val_metrics, self.test_metrics = build_stage_metrics(n)
    ...
    metrics = getattr(self, f"{stage}_metrics")
    metrics(probs, label)                    # updates accumulated state
    self.log_dict(metrics, on_step=True, on_epoch=True, batch_size=batch_size)

Logging the *collection object* (not the returned values) is what lets
Lightning call ``compute()`` once at epoch end.
"""

from __future__ import annotations

from typing import Tuple

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


def build_stage_metrics(
    num_classes: int,
) -> Tuple[MetricCollection, MetricCollection, MetricCollection]:
    """Return independent (train, val, test) metric collections.

    Each stage needs its own instances: sharing one collection would mix
    validation batches into the training accumulation.
    """
    base = MetricCollection(
        {
            "video_acc": MulticlassAccuracy(num_classes=num_classes),
            "video_precision": MulticlassPrecision(num_classes=num_classes),
            "video_recall": MulticlassRecall(num_classes=num_classes),
            "video_f1_score": MulticlassF1Score(num_classes=num_classes),
        }
    )
    return (
        base.clone(prefix="train/"),
        base.clone(prefix="val/"),
        base.clone(prefix="test/"),
    )
