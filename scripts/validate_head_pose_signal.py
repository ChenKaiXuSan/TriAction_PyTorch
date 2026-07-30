#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Offline probe: is head pose alone predictive of the movement label?

Trains a small MLP on the analytic head-pose features (no video, no GPU)
under the person-held-out 5-fold protocol. This gates the expensive model
integration: if these features cannot beat the majority-class baseline on
unseen drivers, wiring them into MV-ViViT is pointless and the keypoint
quality itself is the problem.

Run from the repo root:
    python scripts/validate_head_pose_signal.py [--folds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.models.head_pose_features import FEATURE_DIM, head_pose_features  # noqa: E402

INDEX_DIR = Path("/work/SKIING/chenkaixu/data/drive/index_mapping")
LABELS = ["left", "right", "down", "up"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABELS)}
T_SAMPLES = 8


def _segments_for_video(sample: dict, dataset) -> list:
    """Reuse the dataset's own segment index for one fold split."""
    raise NotImplementedError  # segments come from the dataset below


def _load_head_frames(kpt_dir: Path, wanted: list[int], cache: dict) -> np.ndarray | None:
    """Read pred_keypoints_3d for the nearest available frames to `wanted`."""
    key = str(kpt_dir)
    if key not in cache:
        files = sorted(Path(kpt_dir).glob("*.npz"))
        frames = []
        for f in files:
            tok = f.stem.split("_", 1)[0]
            if tok.isdigit():
                frames.append(int(tok))
        cache[key] = (files, np.asarray(frames))
    files, frames = cache[key]
    if not len(frames):
        return None
    out = []
    for w in wanted:
        i = int(np.abs(frames - w).argmin())
        npz_key = (key, i)
        if npz_key not in cache:
            try:
                with np.load(files[i], allow_pickle=True) as data:
                    cache[npz_key] = np.asarray(
                        data["output"].item()["pred_keypoints_3d"], dtype=np.float32
                    )
            except Exception:
                cache[npz_key] = None
        if cache[npz_key] is None:
            return None
        out.append(cache[npz_key])
    return np.stack(out)


def build_fold_features(fold: int):
    from hydra import compose, initialize_config_dir
    from project.cross_validation import DefineCrossValidation
    from project.main import normalize_dataset_split
    from project.dataloader.data_loader import DriverDataModule

    d = "/work/SKIING/chenkaixu/data/drive"
    overrides = [
        f"paths.root_path={d}",
        f"paths.video_path={d}/videos_split",
        f"paths.sam3d_results_path={d}/sam3d_body_results_right",
        f"paths.start_mid_end_path={d}/annotation/split_mid_end/mini.json",
        "data.split_mode=person_kfold",
        f"data.fold={fold}",
        "train.view=multi",
        "train.view_name=[front,left,right]",
        "model.input_type=rgb",  # rgb only so the dataset does not demand kpt coverage
        "data.num_workers=0",
        "data.val_num_workers=0",
        "data.test_num_workers=0",
    ]
    with initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"), version_base=None
    ):
        cfg = compose(config_name="config", overrides=overrides)
    dm = DriverDataModule(cfg, normalize_dataset_split(DefineCrossValidation(cfg)()))
    dm.setup("fit")

    cache: dict = {}
    out = {}
    for split, ds in (("train", dm.train_gait_dataset), ("val", dm.val_gait_dataset)):
        X, y = [], []
        skipped = 0
        for seg in ds._segment_index:
            item = seg["item"]
            kpt_dir = item.sam3d_kpts.get("front") if item.sam3d_kpts else None
            if kpt_dir is None:
                skipped += 1
                continue
            a, b = int(seg["segment_abs_start"]), int(seg["segment_abs_end"])
            wanted = np.linspace(a, max(a, b - 1), T_SAMPLES).astype(int).tolist()
            arr = _load_head_frames(Path(kpt_dir), wanted, cache)
            if arr is None:
                skipped += 1
                continue
            feats = head_pose_features(torch.from_numpy(arr).unsqueeze(0))[0]
            X.append(feats.flatten().numpy())
            y.append(LABEL_TO_ID[str(seg["label"])])
        out[split] = (np.stack(X), np.asarray(y), skipped)
    return out


def train_probe(Xtr, ytr, Xva, yva, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = torch.from_numpy((Xtr - mu) / sd).float()
    Xva = torch.from_numpy((Xva - mu) / sd).float()
    ytr_t = torch.from_numpy(ytr).long()
    counts = np.bincount(ytr, minlength=4).astype(np.float64)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)), dtype=torch.float32)

    model = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 64),
        torch.nn.GELU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 4),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(Xtr), ytr_t, weight=weights)
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.softmax(model(Xva), dim=1).numpy()
    return probs.argmax(1), probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    all_pred, all_true = [], []
    for fold in args.folds:
        data = build_fold_features(fold)
        Xtr, ytr, sk_tr = data["train"]
        Xva, yva, sk_va = data["val"]
        pred, _ = train_probe(Xtr, ytr, Xva, yva)
        acc = float((pred == yva).mean())
        maj = float(np.bincount(yva, minlength=4).max() / len(yva))
        print(
            f"fold {fold}: n={len(yva)} (skipped tr/va {sk_tr}/{sk_va})  "
            f"probe micro {acc:.3f}  majority {maj:.3f}",
            flush=True,
        )
        all_pred.append(pred)
        all_true.append(yva)

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    from torchmetrics.functional.classification import (
        multiclass_accuracy,
        multiclass_f1_score,
    )

    tp, tt = torch.from_numpy(pred), torch.from_numpy(true)
    micro = float((pred == true).mean())
    maj = float(np.bincount(true, minlength=4).max() / len(true))
    print("\n=== pooled (head-pose features only, unseen drivers) ===")
    print(f"n={len(true)}  micro {micro:.3f}  (majority baseline {maj:.3f})")
    print(f"macro acc {float(multiclass_accuracy(tp, tt, num_classes=4)):.3f}")
    print(f"macro F1  {float(multiclass_f1_score(tp, tt, num_classes=4)):.3f}")
    per = multiclass_f1_score(tp, tt, num_classes=4, average=None)
    print("per-class F1:", {n: round(float(v), 3) for n, v in zip(LABELS, per)})
    counts = Counter(pred.tolist())
    print("prediction distribution:", {LABELS[k]: v for k, v in sorted(counts.items())})


if __name__ == "__main__":
    main()
