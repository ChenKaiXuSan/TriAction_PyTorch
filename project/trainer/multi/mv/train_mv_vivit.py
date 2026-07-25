#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Trainer for MV-ViViT: frozen shared ViViT encoder + cross-view attention fusion."""

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

from project.trainer.losses import build_class_weights, weighted_cross_entropy


class MVViVitTrainer(LightningModule):
    """
    Multi-view classifier with token-level cross-view fusion over ViViT.

    Expected batch format (same as the other multi-view trainers):
        batch["video"][view] : (B, C, T, H, W) for view in train.view_name
        batch["label"]       : (B,)
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        from project.models.mv_vivit import MVViVit

        self.lr = float(hparams.loss.lr)
        self.backbone_lr_scale = float(
            getattr(hparams.model, "mv_vivit_backbone_lr_scale", 0.1)
        )
        self.num_classes = int(hparams.model.model_class_num)
        view_names = getattr(hparams.train, "view_name", ["front", "left", "right"])
        if isinstance(view_names, str):
            view_names = [view_names]
        self.view_names = list(view_names)

        self.model = MVViVit(hparams, view_names=self.view_names)

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
        return self.model(videos)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        videos = {k: v.detach() for k, v in batch["video"].items()}
        label = batch["label"].view(-1)

        logits = self(videos)
        loss = weighted_cross_entropy(logits, label, self.class_weights)

        probs = torch.softmax(logits, dim=1)
        acc = self._accuracy(probs, label)
        precision = self._precision(probs, label)
        recall = self._recall(probs, label)
        f1 = self._f1_score(probs, label)
        _ = self._confusion_matrix(probs, label)

        self.log(
            f"{stage}/loss", loss, on_step=True, on_epoch=True, batch_size=label.size(0)
        )
        self.log_dict(
            {
                f"{stage}/video_acc": acc,
                f"{stage}/video_precision": precision,
                f"{stage}/video_recall": recall,
                f"{stage}/video_f1_score": f1,
            },
            on_step=True,
            on_epoch=True,
            batch_size=label.size(0),
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        # partially unfrozen backbone layers train at a reduced lr
        backbone_params, fusion_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                fusion_params.append(param)
        param_groups = [{"params": fusion_params}]
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": self.lr * self.backbone_lr_scale}
            )
        optimizer = torch.optim.Adam(param_groups, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }
