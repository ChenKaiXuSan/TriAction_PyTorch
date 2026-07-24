#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/main.py
Project: /workspace/code/project
Created Date: Tuesday April 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os
from typing import Any, List, Mapping

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
# os.environ.setdefault('NCCL_SHM_DISABLE', '1')

import hydra
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from project.cross_validation import DefineCrossValidation
from project.dataloader.data_loader import DriverDataModule

#####################################
# select different experiment trainer
#####################################
from project.trainer.multi_selector import build_multi_trainer
from project.trainer.single_selector import build_single_trainer

logger = logging.getLogger(__name__)
RUN_NAME = "run"


def parse_train_devices(gpu_config: Any) -> List[int]:
    """Parse train.gpu while keeping backward compatibility with int configs."""
    if isinstance(gpu_config, int):
        return [gpu_config]

    if isinstance(gpu_config, (list, tuple, ListConfig)):
        return [int(device) for device in gpu_config]

    if isinstance(gpu_config, str):
        value = gpu_config.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        return [int(device.strip()) for device in value.split(",") if device.strip()]

    return [int(gpu_config)]


def normalize_dataset_split(dataset_idx):
    """Return the single train/val split expected by DriverDataModule."""
    if not isinstance(dataset_idx, Mapping):
        raise TypeError("dataset index must be a mapping with train/val splits.")

    if "train" in dataset_idx and "val" in dataset_idx:
        return dataset_idx

    legacy_folds = [
        split
        for split in dataset_idx.values()
        if isinstance(split, Mapping) and "train" in split and "val" in split
    ]
    if len(legacy_folds) == 1 and len(dataset_idx) == 1:
        return legacy_folds[0]

    raise ValueError(
        "Expected a single dataset split with train/val entries. "
        "Multiple folds are no longer supported."
    )


def train(hparams: DictConfig, dataset_idx):
    """Run the training process for a single train/val split.

    Args:
        hparams (hydra): the hyperparameters.
        dataset_idx (dict): the dataset index with train/val splits.

    Returns:
        list: best trained model, data loader
    """

    seed_everything(42, workers=True)

    # * select experiment
    # TODO: add more experiment trainer here.
    if hparams.train.view == "multi":
        classification_module = build_multi_trainer(hparams)
    elif hparams.train.view == "single":
        classification_module = build_single_trainer(hparams)
    else:
        raise ValueError("the experiment view is not supported.")

    # * prepare data module
    data_module = DriverDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.log_path, "tb_logs"),
        name=RUN_NAME,
    )
    csv_logger = CSVLogger(
        save_dir=os.path.join(hparams.log_path, "csv_logs"),
        name=RUN_NAME,
    )

    # some callbacks
    rich_model_summary = RichModelSummary(max_depth=2)
    progress_bar = TQDMProgressBar(refresh_rate=50)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        dirpath=os.path.join(hparams.log_path, "checkpoints", RUN_NAME),
        filename="{epoch}-{val/loss:.2f}",
        auto_insert_metric_name=False,
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=5,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=parse_train_devices(hparams.train.gpu),
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        precision=getattr(hparams.train, "precision", "32-true"),
        accumulate_grad_batches=int(
            getattr(hparams.train, "accumulate_grad_batches", 1)
        ),
        limit_train_batches=getattr(hparams.train, "limit_train_batches", 1.0),
        limit_val_batches=getattr(hparams.train, "limit_val_batches", 1.0),
        limit_test_batches=getattr(hparams.train, "limit_test_batches", 1.0),
        logger=[tb_logger, csv_logger],
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            early_stopping,
            lr_monitor,
            DeviceStatsMonitor(),  # monitor the device stats.
        ],
    )

    trainer.fit(classification_module, data_module)

    # save the metrics to file
    trainer.test(
        classification_module,
        data_module,
        ckpt_path="best",
        weights_only=False,
    )


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="config.yaml",
)
def init_params(config):
    #######################
    # prepare dataset index
    #######################

    dataset_idx = normalize_dataset_split(DefineCrossValidation(config)())

    logger.info("#" * 50)
    logger.info("Start train")
    logger.info("#" * 50)

    train(config, dataset_idx)

    logger.info("#" * 50)
    logger.info("finish train")
    logger.info("#" * 50)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
