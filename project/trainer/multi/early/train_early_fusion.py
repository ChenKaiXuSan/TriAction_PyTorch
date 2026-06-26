#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/train_late_fusion.py
Project: /workspace/skeleton/project
Created Date: Monday May 13th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday October 28th 2025 9:49:19 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, Dict

import torch

from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from project.models.make_model import select_model
from project.trainer.losses import build_class_weights, weighted_cross_entropy


class EarlyFusion3DCNNTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        self.lr = float(getattr(hparams.loss, "lr", 1e-3))
        self.num_classes = hparams.model.model_class_num
        self.input_type = getattr(hparams.model, "input_type", "rgb")
        self.fuse_method = getattr(hparams.model, "fuse_method", "avg")
        self.view_names = getattr(hparams.train, "view_name", ["front", "left", "right"])
        if isinstance(self.view_names, str):
            self.view_names = [self.view_names]
        self.view_names = list(self.view_names)
        self.num_views = len(self.view_names)

        # define model
        self.view_cnns = torch.nn.ModuleDict(
            {view: select_model(hparams) for view in self.view_names}
        )
        self.view_fusion_head = None
        if self.fuse_method == "concat":
            feature_dim = self._infer_feature_dim(next(iter(self.view_cnns.values())))
            self.view_fusion_head = torch.nn.Linear(
                feature_dim * self.num_views, self.num_classes
            )

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        class_weights = build_class_weights(hparams)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, videos: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(videos, dict):
            raise TypeError("Early fusion expects batch['video'] to be a dict of views.")

        if self.fuse_method == "concat":
            features = []
            for view in self.view_names:
                model = self.view_cnns[view]
                if not hasattr(model, "forward_features"):
                    raise ValueError(
                        f"Selected model {type(model).__name__} does not support concat fusion."
                    )
                features.append(model.forward_features(videos[view]))
            return self.view_fusion_head(torch.cat(features, dim=1))

        logits = [self.view_cnns[view](videos[view]) for view in self.view_names]
        stacked_logits = torch.stack(logits, dim=0)
        if self.fuse_method == "add":
            return stacked_logits.sum(dim=0)
        if self.fuse_method == "avg":
            return stacked_logits.mean(dim=0)
        if self.fuse_method == "mul":
            probs = torch.softmax(stacked_logits, dim=-1).prod(dim=0)
            return torch.log(torch.clamp(probs, min=1e-8))
        raise ValueError(f"Unknown early-fusion method: {self.fuse_method}")

    @staticmethod
    def _infer_feature_dim(model) -> int:
        feature_dim = getattr(model, "feature_dim", None)
        if feature_dim is None:
            raise ValueError(
                f"Selected model {type(model).__name__} lacks feature_dim for concat fusion."
            )
        return int(feature_dim)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        videos = batch["video"]
        if videos is None:
            raise ValueError("RGB videos are required for early fusion.")
        videos = {view: videos[view].detach() for view in self.view_names}
        label = batch["label"].view(-1)

        logits = self(videos)
        loss = weighted_cross_entropy(logits, label, self.class_weights)
        probs = torch.softmax(logits, dim=1)

        video_acc = self._accuracy(probs, label)
        video_precision = self._precision(probs, label)
        video_recall = self._recall(probs, label)
        video_f1_score = self._f1_score(probs, label)
        _ = self._confusion_matrix(probs, label)

        self.log(
            f"{stage}/loss", loss, on_epoch=True, on_step=True, batch_size=label.size(0)
        )
        self.log_dict(
            {
                f"{stage}/video_acc": video_acc,
                f"{stage}/video_precision": video_precision,
                f"{stage}/video_recall": video_recall,
                f"{stage}/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=label.size(0),
        )

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.estimated_stepping_batches,
                    verbose=True,
                ),
                "monitor": "train/loss",
            },
        }


class EarlyFusionTransformerTrainer(EarlyFusion3DCNNTrainer):
    """Early fusion trainer alias for transformer backbone routing."""


class EarlyFusionMambaTrainer(EarlyFusion3DCNNTrainer):
    """Early fusion trainer alias for mamba backbone routing."""


class EarlyFusionSTGCNTrainer(EarlyFusion3DCNNTrainer):
    """Early fusion trainer alias for ST-GCN backbone routing."""


class EarlyFusionRGBKeypointTrainer(EarlyFusion3DCNNTrainer):
    """Early fusion trainer alias for RGB+KPT fusion backbone."""
