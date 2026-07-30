#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Logit-level fusion of the head-pose probe with a video model.

The probe and the video models fail in complementary ways: the probe makes
non-trivial predictions on right/up (where every video model is near zero)
while the video models dominate on left/down. Token-level fusion inside
MV-ViViT kept the accuracy gain but lost the probe's rare-class ability, so
this fuses at the logit level instead, where neither side can silence the
other.

No tuning happens on the reported fold: the primary fusion is a fixed 50/50
probability average. The alpha sweep is printed as a sensitivity curve only
-- picking its best point post hoc would be test-set selection.

Alignment safety: the probe walks the same segment index that the test
dataloader used (shuffle=False, drop_last=False), and every fold asserts that
the probe's label sequence is identical to the one saved next to the model
predictions before any fusion is computed.

Run from the repo root:
    python scripts/logit_fusion.py --arm K_headpose
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torchmetrics.functional.classification import multiclass_f1_score  # noqa: E402

from validate_head_pose_signal import LABELS, build_fold_features, train_probe  # noqa: E402

LOG_ROOT = Path("logs/train")


def _model_preds(arm: str, fold: int):
    d = sorted(LOG_ROOT.glob(f"{arm}_fold{fold}/*/*/best_preds"))[-1]
    pred = torch.load(d / "run_pred.pt", map_location="cpu").float()
    label = torch.load(d / "run_label.pt", map_location="cpu")
    return pred.numpy(), label.numpy()


def _report(tag: str, probs: np.ndarray, y: np.ndarray) -> None:
    pred = probs.argmax(1)
    tp, tt = torch.from_numpy(pred), torch.from_numpy(y)
    per = multiclass_f1_score(tp, tt, num_classes=4, average=None)
    print(
        f"{tag:28s} micro {float((pred == y).mean()):.3f}  "
        f"macroF1 {float(multiclass_f1_score(tp, tt, num_classes=4)):.3f}  "
        + "  ".join(f"{n}:{float(v):.2f}" for n, v in zip(LABELS, per))
    )


def _mcnemar(pa: np.ndarray, pb: np.ndarray, y: np.ndarray) -> float:
    ca, cb = pa.argmax(1) == y, pb.argmax(1) == y
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    if n01 + n10 == 0:
        return 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return math.erfc(math.sqrt(chi2 / 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="K_headpose", help="video-model K-fold arm")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    model_all, probe_all, y_all = [], [], []
    for fold in args.folds:
        data = build_fold_features(fold)
        Xtr, ytr, _ = data["train"]
        Xva, yva, _ = data["val"]
        _, probe_probs = train_probe(Xtr, ytr, Xva, yva)

        model_probs, model_y = _model_preds(args.arm, fold)
        if not np.array_equal(model_y, yva):
            raise RuntimeError(
                f"fold {fold}: probe and {args.arm} label sequences differ "
                f"({len(yva)} vs {len(model_y)}) — segment alignment broken, "
                "fusion would be meaningless."
            )
        model_all.append(model_probs)
        probe_all.append(probe_probs)
        y_all.append(yva)
        print(f"fold {fold}: aligned n={len(yva)}", flush=True)

    model = np.concatenate(model_all)
    probe = np.concatenate(probe_all)
    y = np.concatenate(y_all)

    print(f"\n=== pooled fusion with {args.arm} (n={len(y)}) ===")
    _report("video model alone", model, y)
    _report("head-pose probe alone", probe, y)
    fused = (model + probe) / 2.0
    _report("fusion 50/50 (primary)", fused, y)
    print(
        f"fusion vs model alone: McNemar p={_mcnemar(model, fused, y):.2g}, "
        f"delta micro {float((fused.argmax(1) == y).mean() - (model.argmax(1) == y).mean()):+.3f}"
    )

    print("\nalpha sensitivity (alpha * model + (1-alpha) * probe), no selection:")
    for alpha in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        _report(f"  alpha={alpha:.1f}", alpha * model + (1 - alpha) * probe, y)


if __name__ == "__main__":
    main()
