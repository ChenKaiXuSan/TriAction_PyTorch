#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Aggregate person-wise K-fold runs into one cross-validated score.

Every fold validates on persons the model never trained on, so pooling the
folds' saved predictions scores each person exactly once. Report the pooled
number as the headline and the per-fold spread as the uncertainty -- a single
split of this dataset carries a +/-6% confidence interval, which is wider than
every architectural difference measured so far.

Usage:
    python scripts/aggregate_folds.py K_mvvivit K_baseline3dcnn
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
)

LOG_ROOT = Path("logs/train")
NUM_CLASSES = 4


def _load_fold(exp: str):
    """Newest saved prediction/label pair for one experiment, or None."""
    dirs = sorted(LOG_ROOT.glob(f"{exp}/*/*/best_preds"))
    for d in reversed(dirs):
        pred_p, label_p = d / "run_pred.pt", d / "run_label.pt"
        if not (pred_p.exists() and label_p.exists()):
            continue
        pred = torch.load(pred_p, map_location="cpu").float()
        label = torch.load(label_p, map_location="cpu")
        if pred.shape[0] == label.shape[0] and label.numel():
            return pred, label
    return None


def _scores(pred: torch.Tensor, label: torch.Tensor) -> dict:
    return {
        "micro": float((pred.argmax(1) == label).float().mean()),
        "macro": float(multiclass_accuracy(pred, label, num_classes=NUM_CLASSES)),
        "macro_f1": float(multiclass_f1_score(pred, label, num_classes=NUM_CLASSES)),
    }


def aggregate(prefix: str, num_folds: int = 5) -> None:
    folds, missing = [], []
    for f in range(num_folds):
        loaded = _load_fold(f"{prefix}_fold{f}")
        (folds.append((f, *loaded)) if loaded else missing.append(f))

    print(f"\n=== {prefix} ===")
    if not folds:
        print("  no completed folds found")
        return
    if missing:
        print(f"  !! folds not available: {missing} (numbers below cover only {len(folds)}/{num_folds})")

    for f, pred, label in folds:
        s = _scores(pred, label)
        print(
            f"  fold {f}: n={len(label):4d}  micro {s['micro']:.3f}  "
            f"macro {s['macro']:.3f}  macroF1 {s['macro_f1']:.3f}"
        )

    pooled_pred = torch.cat([p for _, p, _ in folds])
    pooled_label = torch.cat([l for _, _, l in folds])
    pooled = _scores(pooled_pred, pooled_label)
    per_fold = [_scores(p, l)["micro"] for _, p, l in folds]
    spread = (
        math.sqrt(sum((x - sum(per_fold) / len(per_fold)) ** 2 for x in per_fold) / (len(per_fold) - 1))
        if len(per_fold) > 1
        else 0.0
    )
    n = len(pooled_label)
    ci = 1.96 * math.sqrt(pooled["micro"] * (1 - pooled["micro"]) / n)
    print(
        f"  pooled (n={n}): micro {pooled['micro']:.3f} +/-{ci:.3f} (95% CI)  "
        f"macro {pooled['macro']:.3f}  macroF1 {pooled['macro_f1']:.3f}"
    )
    print(f"  per-fold micro spread (std): {spread:.3f}")


if __name__ == "__main__":
    prefixes = sys.argv[1:] or ["K_mvvivit", "K_baseline3dcnn"]
    for prefix in prefixes:
        aggregate(prefix)
