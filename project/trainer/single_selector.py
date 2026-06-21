#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Single-view training selection helpers.
"""

import logging

from project.trainer.single.train_single_modality import SingleModalityClassifierTrainer

logger = logging.getLogger(__name__)


SUPPORTED_RGB_BACKBONES = {"3dcnn", "transformer", "mamba"}
SUPPORTED_INPUT_TYPES = {"rgb", "kpt", "rgb_kpt"}


def select_single_trainer_cls(hparams):
    """Select the trainer class for configurable modality/view experiments."""
    if getattr(hparams.train, "view", None) != "single":
        raise ValueError("Single-view trainer only supports train.view=single.")

    input_type = getattr(hparams.model, "input_type", "rgb")
    if input_type not in SUPPORTED_INPUT_TYPES:
        raise ValueError(
            f"input_type {input_type} is not supported. "
            f"Supported input types: {sorted(SUPPORTED_INPUT_TYPES)}"
        )

    backbone = getattr(hparams.model, "backbone", None)
    if input_type in {"rgb", "rgb_kpt"} and backbone not in SUPPORTED_RGB_BACKBONES:
        raise ValueError(
            f"backbone {backbone} is not supported for single-view RGB training."
        )

    return SingleModalityClassifierTrainer


def build_single_trainer(hparams):
    trainer_cls = select_single_trainer_cls(hparams)
    return trainer_cls(hparams)
