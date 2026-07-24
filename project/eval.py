#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/eval.py
Project: /workspace/code/project
Created Date: Tuesday November 11th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday November 11th 2025 1:34:34 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import re
import glob
import json
import time
import math
import logging
from typing import Dict, List, Tuple, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import DeviceStatsMonitor

# DataModule
from dataloader.data_loader import DriverDataModule

# Trainers (LightningModules)
from trainer.baseline.train_3dcnn import Res3DCNNTrainer
from trainer.mid.train_pose_attn import PoseAttnTrainer
from trainer.early.train_early_fusion import EarlyFusion3DCNNTrainer
from trainer.late.train_late_fusion import LateFusion3DCNNTrainer

# Dataset splitter
from cross_validation import DefineCrossValidation
from main import normalize_dataset_split, RUN_NAME

logger = logging.getLogger(__name__)


def _cfg_get(config: DictConfig, path: str, default=None):
    value = OmegaConf.select(config, path)
    return default if value is None else value


def _select_module(hparams: DictConfig):
    """Mirror the selection logic used in main.py"""
    if hparams.model.backbone != "3dcnn":
        raise ValueError("Only backbone='3dcnn' is supported in this eval script.")

    fm = hparams.model.fuse_method
    if fm == "pose_atn":
        return PoseAttnTrainer(hparams)
    elif fm in ["add", "mul", "concat", "avg"]:
        return EarlyFusion3DCNNTrainer(hparams)
    elif fm == "late":
        return LateFusion3DCNNTrainer(hparams)
    elif fm == "none":
        return Res3DCNNTrainer(hparams)
    else:
        raise ValueError(f"Unsupported fuse_method: {fm}")


def _parse_ckpt_metric(path: str) -> Optional[Tuple[int, float, float]]:
    """Parse epoch, val_loss, val_acc from checkpoint filename."""

    base = os.path.basename(path)
    epoch, vloss, vacc = base.split("-")[0:3]
    vacc = vacc.replace(".ckpt", "")

    return int(epoch), float(vloss), float(vacc)


def _find_best_ckpt(log_path: str) -> Optional[str]:
    """
    Search checkpoints for the single run and pick the highest val/video_acc.
    Fallback order:
      1) best by parsed val/video_acc
      2) last.ckpt
      3) None (caller will run without pretrained weights)
    """
    # 1) collect all candidate ckpts with metrics in filename
    candidates = glob.glob(os.path.join(log_path, "checkpoints", RUN_NAME, "*.ckpt"))
    if not candidates:
        candidates = glob.glob(os.path.join(log_path, "checkpoints", "fold_*", "*.ckpt"))
    best_path = None
    best_acc = -math.inf

    for p in candidates:
        if "last.ckpt" in os.path.basename(p).lower():
            continue
        parsed = _parse_ckpt_metric(p)
        if parsed is None:
            continue
        _, _, vacc = parsed
        if vacc > best_acc:
            best_acc = vacc
            best_path = p

    if best_path is not None:
        return best_path

    # 2) try last.ckpt
    last_candidates = [
        p for p in candidates if os.path.basename(p).lower() == "last.ckpt"
    ]
    if last_candidates:
        # if multiple, choose the newest by mtime
        last_candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return last_candidates[0]

    # 3) nothing found
    return None


def _aggregate(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate a list of test result dicts.
    Returns a dict: metric -> {"mean": ..., "std": ...}
    Only aggregates keys starting with 'test/' and whose values are numbers.
    """
    from collections import defaultdict
    import numpy as np

    buckets = defaultdict(list)
    for r in results:
        for k, v in r.items():
            if isinstance(v, (int, float)) and k.startswith("test/"):
                buckets[k].append(float(v))

    agg: Dict[str, Dict[str, float]] = {}
    for k, arr in buckets.items():
        if len(arr) == 0:
            continue
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=0))
        agg[k] = {"mean": m, "std": s}
    return agg


def _save_outputs(
    out_dir: str,
    run_results: Dict[str, Dict[str, float]],
    aggregate_stats: Dict[str, Dict[str, float]],
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(out_dir, f"eval_results_{ts}.json")
    csv_path = os.path.join(out_dir, f"eval_results_{ts}.csv")

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": run_results,
                "aggregate": aggregate_stats,
                "created_at": ts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # CSV (flatten)
    import csv

    # collect header
    metric_names = set()
    for _run, metrics in run_results.items():
        metric_names.update([k for k in metrics.keys() if k.startswith("test/")])
    metric_names = sorted(metric_names)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run"] + metric_names)
        for run_name, metrics in sorted(run_results.items(), key=lambda x: str(x[0])):
            row = [run_name] + [metrics.get(m, "") for m in metric_names]
            writer.writerow(row)
        # add a blank line and aggregate
        writer.writerow([])
        writer.writerow(["metric", "mean", "std"])
        for m in metric_names:
            ms = aggregate_stats.get(m, {})
            writer.writerow([m, ms.get("mean", ""), ms.get("std", "")])

    return json_path, csv_path


def _make_trainer(hparams: DictConfig) -> Trainer:
    """Minimal trainer for evaluation."""
    return Trainer(
        devices=[int(hparams.train.gpu)],
        accelerator="gpu",
        logger=CSVLogger(
            save_dir=os.path.join(hparams.train.log_path, "eval_csv_logs"),
            name="eval",
        ),
        callbacks=[DeviceStatsMonitor()],
    )


def _eval_one_run(hparams: DictConfig, dataset_idx) -> Dict[str, float]:
    """Run test() for the single split and return the metrics dict."""
    seed_everything(42, workers=True)

    # module and data
    module = _select_module(hparams)
    datamodule = DriverDataModule(hparams, dataset_idx)

    # locate ckpt
    ckpt_root = _cfg_get(hparams, "eval.input_path", _cfg_get(hparams, "log_path"))
    if ckpt_root is None:
        raise ValueError("Missing eval.input_path or log_path for checkpoint lookup.")
    ckpt = _find_best_ckpt(str(ckpt_root))
    if ckpt:
        logger.info(f"Using checkpoint: {ckpt}")
    else:
        logger.warning(
            "No checkpoint found. Running test() with randomly initialized weights."
        )

    trainer = _make_trainer(hparams)

    # Run test
    if ckpt:
        test_out = trainer.test(module, datamodule, ckpt_path=ckpt)
    else:
        test_out = trainer.test(module, datamodule)

    # PL returns a list[dict]; typically len==1 unless multiple test loaders
    if not test_out:
        logger.warning("Empty test result; returning empty dict.")
        return {}

    # If multiple dicts, merge keys by later overwriting (usually fine)
    merged: Dict[str, float] = {}
    for d in test_out:
        merged.update(d)
    return merged


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config.yaml",
)
def main(config: DictConfig):
    """
    Single-split evaluation:
    - Load one train/validation split using DefineCrossValidation(config)()
    - Load best checkpoint if available and run trainer.test
    - Save run results and aggregate mean/std to log_path
    """
    dataset_idx = normalize_dataset_split(DefineCrossValidation(config)())

    logger.info("#" * 60)
    logger.info("Start EVALUATION")
    logger.info("#" * 60)

    metrics = _eval_one_run(config, dataset_idx)
    run_results = {RUN_NAME: metrics}
    if metrics:
        nice = {
            k: round(v, 6)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        logger.info("test metrics: %s", nice)
    else:
        logger.info("No metrics returned.")

    # Aggregate
    aggregate_stats = _aggregate(list(run_results.values()))

    logger.info("#" * 60)
    logger.info("Aggregate (mean ± std) for test/* metrics:")
    for m, s in sorted(aggregate_stats.items()):
        logger.info(f"  {m}: mean={s['mean']:.6f}, std={s['std']:.6f}")
    logger.info("#" * 60)

    # Save
    out_dir = _cfg_get(config, "eval.log_path", _cfg_get(config, "log_path"))
    if out_dir is None:
        raise ValueError("Missing eval.log_path or log_path for evaluation outputs.")
    json_path, csv_path = _save_outputs(out_dir, run_results, aggregate_stats)
    logger.info(f"Saved evaluation results:\n  JSON: {json_path}\n  CSV : {csv_path}")
    logger.info("Finished EVALUATION.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
