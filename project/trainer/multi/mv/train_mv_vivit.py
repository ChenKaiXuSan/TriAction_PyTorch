#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Trainer for MV-ViViT: frozen shared ViViT encoder + cross-view attention fusion."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from project.trainer.metrics import build_stage_metrics
from project.trainer.losses import build_class_weights, classification_loss
from project.utils.helper import save_helper

# raw SAM-3D rows: nose / left-eye / right-eye / left-ear / right-ear
_NOSE, _LEYE, _REYE, _LEAR, _REAR = 0, 1, 2, 3, 4


def head_pose_targets(kpts: torch.Tensor, steps: int) -> torch.Tensor:
    """(B, T, K, 3) head keypoints -> (B, 2*steps) yaw/pitch trajectory.

    Facing direction is approximated by the vector from the ear midpoint to the
    nose; yaw/pitch conventions only need to be self-consistent since this is
    an auxiliary regression target.
    """
    ears_mid = (kpts[:, :, _LEAR, :] + kpts[:, :, _REAR, :]) / 2.0
    facing = kpts[:, :, _NOSE, :] - ears_mid  # (B, T, 3)
    yaw = torch.atan2(facing[..., 0], facing[..., 2].abs() + 1e-6)
    pitch = torch.atan2(
        facing[..., 1], facing[..., [0, 2]].norm(dim=-1) + 1e-6
    )
    traj = torch.stack([yaw, pitch], dim=1)  # (B, 2, T)
    traj = torch.nn.functional.adaptive_avg_pool1d(traj, steps)
    return traj.flatten(1)


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
        self._loss_hparams = hparams
        self.save_root = getattr(hparams, "log_path", "./logs")
        self.llrd_gamma = float(getattr(hparams.model, "mv_vivit_llrd_gamma", 1.0))
        self.aux_pose_weight = float(
            getattr(hparams.model, "mv_vivit_aux_pose_weight", 0.0)
        )
        self.test_pred_list: list = []
        self.test_label_list: list = []

        # accumulated per stage and reduced once per epoch (see project/trainer/metrics.py)
        self.train_metrics, self.val_metrics, self.test_metrics = build_stage_metrics(
            self.num_classes
        )
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        class_weights = build_class_weights(hparams)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        videos: Dict[str, torch.Tensor],
        kpts: Optional[Dict[str, torch.Tensor]] = None,
        head_videos: Optional[Dict[str, torch.Tensor]] = None,
        return_pooled: bool = False,
    ):
        return self.model(
            videos, kpts=kpts, head_videos=head_videos, return_pooled=return_pooled
        )

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        videos = {k: v.detach() for k, v in batch["video"].items()}
        kpts = batch.get("sam3d_kpt")
        if kpts is not None:
            kpts = {k: v.detach() for k, v in kpts.items() if v is not None} or None
        head_videos = batch.get("head_video")
        if head_videos is not None:
            head_videos = {k: v.detach() for k, v in head_videos.items()}
        label = batch["label"].view(-1)

        use_aux = self.aux_pose_weight > 0 and self.model.aux_pose_head is not None
        if use_aux:
            logits, pooled = self(videos, kpts, head_videos, return_pooled=True)
        else:
            logits = self(videos, kpts, head_videos)
        loss = classification_loss(
            logits, label, self.class_weights, hparams=self._loss_hparams
        )
        if use_aux:
            if kpts is None or not kpts:
                raise ValueError(
                    "mv_vivit_aux_pose_weight>0 requires keypoints "
                    "(set model.input_type=rgb_kpt)."
                )
            ref_view = "front" if "front" in kpts else next(iter(kpts))
            target = head_pose_targets(
                kpts[ref_view].float(), self.model.aux_pose_steps
            )
            aux_pred = self.model.aux_pose_head(pooled)
            aux_loss = torch.nn.functional.smooth_l1_loss(aux_pred, target)
            self.log(
                f"{stage}/aux_pose_loss",
                aux_loss,
                on_step=False,
                on_epoch=True,
                batch_size=label.size(0),
            )
            loss = loss + self.aux_pose_weight * aux_loss

        probs = torch.softmax(logits, dim=1)

        self.log(
            f"{stage}/loss", loss, on_step=True, on_epoch=True, batch_size=label.size(0)
        )
        metrics = getattr(self, f"{stage}_metrics")
        metrics(probs, label)
        self.log_dict(
            metrics, on_step=True, on_epoch=True, batch_size=label.size(0)
        )
        if stage == "test":
            self.test_pred_list.append(probs.detach().cpu())
            self.test_label_list.append(label.detach().cpu())
        return loss

    def on_test_start(self) -> None:
        self.test_pred_list = []
        self.test_label_list = []

    def on_test_epoch_end(self) -> None:
        if not self.test_pred_list or not self.test_label_list:
            return
        fold = "run"
        if self.logger and getattr(self.logger, "root_dir", None):
            fold = Path(self.logger.root_dir).name
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=fold,
            save_path=self.save_root,
            num_class=self.num_classes,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        # partially unfrozen backbone layers train at a reduced lr; with
        # llrd_gamma < 1 each layer below the top decays further (layer-wise
        # lr decay)
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and not n.startswith("backbone.")
                ]
            }
        ]
        layers = self.model._encoder_layers(self.model.backbone)
        trainable_layers = [
            layer
            for layer in layers
            if any(p.requires_grad for p in layer.parameters())
        ]
        grouped = set()
        for depth, layer in enumerate(reversed(trainable_layers)):
            layer_params = [p for p in layer.parameters() if p.requires_grad]
            grouped.update(id(p) for p in layer_params)
            param_groups.append(
                {
                    "params": layer_params,
                    "lr": self.lr * self.backbone_lr_scale * (self.llrd_gamma**depth),
                }
            )
        leftovers = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and n.startswith("backbone.") and id(p) not in grouped
        ]
        if leftovers:
            param_groups.append(
                {"params": leftovers, "lr": self.lr * self.backbone_lr_scale}
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
