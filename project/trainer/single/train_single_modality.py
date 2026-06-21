#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Single/multi-selected-view classification trainer for RGB, KPT, and RGB+KPT."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from project.models.make_model import select_model
from project.trainer.losses import build_class_weights, weighted_cross_entropy
from project.utils.helper import save_helper

logger = logging.getLogger(__name__)


class SingleModalityClassifierTrainer(LightningModule):
    """Classification trainer with configurable modality and view list."""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(getattr(hparams.loss, "lr", 1e-3))
        self.num_classes = int(hparams.model.model_class_num)
        self.input_type = getattr(hparams.model, "input_type", "rgb")
        self.view_names = getattr(hparams.train, "view_name", ["front"])
        if isinstance(self.view_names, str):
            self.view_names = [self.view_names]
        if not self.view_names:
            raise ValueError("train.view_name must contain at least one view.")

        self.model = select_model(hparams)
        self.save_root = getattr(hparams, "log_path", "./logs")

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

        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []

    def forward(
        self,
        video: Optional[torch.Tensor] = None,
        kpts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.input_type == "rgb":
            return self.model(video)
        if self.input_type == "kpt":
            return self.model(kpts)
        if self.input_type == "rgb_kpt":
            return self.model(video, kpts)
        raise ValueError(f"Unknown input_type: {self.input_type}")

    def _get_view_tensor(
        self,
        data: Optional[Dict[str, torch.Tensor]],
        view: str,
        required: bool,
        name: str,
    ) -> Optional[torch.Tensor]:
        if data is None:
            if required:
                raise ValueError(f"{name} input requested but batch['{name}'] is missing.")
            return None
        if view not in data or data[view] is None:
            if required:
                raise KeyError(f"View '{view}' not found in batch['{name}'].")
            return None
        return data[view].detach()

    def _forward_selected_views(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits = []
        need_rgb = self.input_type in {"rgb", "rgb_kpt"}
        need_kpt = self.input_type in {"kpt", "rgb_kpt"}

        for view in self.view_names:
            video = self._get_view_tensor(batch.get("video"), view, need_rgb, "video")
            kpts = self._get_view_tensor(batch.get("sam3d_kpt"), view, need_kpt, "sam3d_kpt")
            logits.append(self(video, kpts))

        return torch.stack(logits, dim=0).mean(dim=0)

    @staticmethod
    def _prepare_label(batch: Dict[str, Any]) -> torch.Tensor:
        return batch["label"].detach().view(-1)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        label = self._prepare_label(batch)
        logits = self._forward_selected_views(batch)
        probs = torch.softmax(logits, dim=1)
        loss = weighted_cross_entropy(logits, label, self.class_weights)

        batch_size = int(label.shape[0])
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=True, batch_size=batch_size)

        acc = self._accuracy(probs, label)
        precision = self._precision(probs, label)
        recall = self._recall(probs, label)
        f1_score = self._f1_score(probs, label)
        _ = self._confusion_matrix(probs, label)

        self.log_dict(
            {
                f"{stage}/video_acc": acc,
                f"{stage}/video_precision": precision,
                f"{stage}/video_recall": recall,
                f"{stage}/video_f1_score": f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
        )

        if stage == "test":
            self.test_pred_list.append(probs.detach().cpu())
            self.test_label_list.append(label.detach().cpu())

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, "val")

    def on_test_start(self) -> None:
        self.test_pred_list = []
        self.test_label_list = []

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        if not self.test_pred_list or not self.test_label_list:
            return
        fold = "fold"
        if self.logger and getattr(self.logger, "root_dir", None):
            fold = Path(self.logger.root_dir).name
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=fold,
            save_path=self.save_root,
            num_class=self.num_classes,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.estimated_stepping_batches,
                ),
                "monitor": "train/loss",
            },
        }
