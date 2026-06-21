#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/models/make_model.py
Project: /workspace/skeleton/project/models
Created Date: Thursday October 19th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday April 19th 2025 7:58:58 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

26-11-2024	Kaixu Chen	remove x3d network.
"""

# ! prepare not used
import torch.nn as nn

from project.models.keypoint_classifier import KeypointTemporalClassifier, RGBKPTClassifier


def select_rgb_model(hparams) -> nn.Module:
    model_backbone = hparams.model.backbone

    if model_backbone == "3dcnn":
        from project.models.res_3dcnn import Res3DCNN

        model = Res3DCNN(hparams)
    elif model_backbone == "transformer":
        from project.models.video_transformer import VideoTransformer

        model = VideoTransformer(hparams)
    elif model_backbone == "mamba":
        from project.models.video_mamba import VideoMamba

        model = VideoMamba(hparams)
    else:
        raise ValueError(f"Unknown model backbone: {model_backbone}")

    return model


def select_model(hparams) -> nn.Module:
    """Select a classification model based on ``model.input_type``."""

    input_type = getattr(hparams.model, "input_type", "rgb")

    if input_type == "rgb":
        return select_rgb_model(hparams)
    if input_type == "kpt":
        return KeypointTemporalClassifier(hparams)
    if input_type == "rgb_kpt":
        return RGBKPTClassifier(
            rgb_model=select_rgb_model(hparams),
            kpt_model=KeypointTemporalClassifier(hparams),
        )

    raise ValueError(f"Unknown model input_type: {input_type}")
