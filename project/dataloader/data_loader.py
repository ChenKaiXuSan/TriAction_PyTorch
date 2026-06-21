#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/MultiView_DriverAction_PyTorch/project/dataloader/data_loader.py
Project: /workspace/MultiView_DriverAction_PyTorch/project/dataloader
Created Date: Saturday January 24th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday January 24th 2026 10:51:04 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import random
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Resize,
)

from project.dataloader.whole_video_dataset import whole_video_dataset
from project.dataloader.annotation_dict import get_annotation_dict
from project.dataloader.utils import (
    Div255,
    UniformTemporalSubsample,
)


class SegmentGroupedSampler(Sampler[int]):
    """Shuffle segment chunks while keeping segments from the same chunk adjacent."""

    def __init__(self, dataset, seed: int = 0, shuffle: bool = True):
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.groups: List[List[int]] = []

        current_key = None
        current_group: List[int] = []
        for index, segment in enumerate(getattr(dataset, "_segment_index", [])):
            key = segment.get("source_index")
            if current_key is None or key == current_key:
                current_group.append(index)
            else:
                self.groups.append(current_group)
                current_group = [index]
            current_key = key
        if current_group:
            self.groups.append(current_group)

    def __iter__(self) -> Iterator[int]:
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(groups)
        self.epoch += 1
        for group in groups:
            yield from group

    def __len__(self) -> int:
        return len(self.dataset)


class DriverDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._num_workers = int(opt.data.num_workers)
        self._val_num_workers = int(
            getattr(opt.data, "val_num_workers", self._num_workers)
        )
        self._test_num_workers = int(
            getattr(opt.data, "test_num_workers", self._val_num_workers)
        )
        self._prefetch_factor = int(getattr(opt.data, "prefetch_factor", 1))
        self._img_size = opt.data.img_size

        # frame rate
        self.uniform_temporal_subsample_num = opt.data.uniform_temporal_subsample_num

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx

        self._class_num = opt.model.model_class_num

        self._experiment = opt.experiment

        # * new config paths for annotation
        self._annotation_file = opt.paths.start_mid_end_path

        self._batch_size = opt.data.batch_size
        self.max_video_frames = opt.data.max_video_frames
        self.batch_unit = getattr(opt.data, "batch_unit", "chunk")
        self.segment_grouped_shuffle = bool(
            getattr(opt.data, "segment_grouped_shuffle", True)
        )
        self.segment_shuffle_seed = int(getattr(opt.data, "segment_shuffle_seed", 42))
        self.view_name = opt.train.view_name
        if isinstance(self.view_name, str):
            self.view_name = [self.view_name]

        self.input_type = getattr(opt.model, "input_type", "rgb")
        if self.input_type not in {"rgb", "kpt", "rgb_kpt"}:
            raise ValueError(f"Unsupported model.input_type: {self.input_type}")
        self.load_rgb = self.input_type in {"rgb", "rgb_kpt"}
        self.load_kpt = self.input_type in {"kpt", "rgb_kpt"}

        self.mapping_transform = Compose(
            [
                UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                Div255(),
                Resize(size=[self._img_size, self._img_size]),
            ]
        )

    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        # * lazy load annotation dict from config
        _annotation_dict = get_annotation_dict(self._annotation_file)

        # train dataset
        self.train_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["train"],
            annotation_dict=_annotation_dict,
            transform=self.mapping_transform,
            max_video_frames=self.max_video_frames,
            batch_unit=self.batch_unit,
            view_name=self.view_name,
            load_rgb=self.load_rgb,
            load_kpt=self.load_kpt,
            kpt_temporal_subsample_num=self.uniform_temporal_subsample_num,
        )

        # val dataset
        self.val_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["val"],
            annotation_dict=_annotation_dict,
            transform=self.mapping_transform,
            max_video_frames=self.max_video_frames,
            batch_unit=self.batch_unit,
            view_name=self.view_name,
            load_rgb=self.load_rgb,
            load_kpt=self.load_kpt,
            kpt_temporal_subsample_num=self.uniform_temporal_subsample_num,
        )

        # test dataset
        self.test_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["val"],
            annotation_dict=_annotation_dict,
            transform=self.mapping_transform,
            max_video_frames=self.max_video_frames,
            batch_unit=self.batch_unit,
            view_name=self.view_name,
            load_rgb=self.load_rgb,
            load_kpt=self.load_kpt,
            kpt_temporal_subsample_num=self.uniform_temporal_subsample_num,
        )

    def _collate_fn(self, batch: Any) -> Any:
        # 这里需要在返回batch的时候，把batch和chunk结合起来
        # Merge per-sample segments into one batch dimension.
        if not batch:
            return {}

        views = self.view_name
        video_lists = {view: [] for view in views}
        kpt_lists = {view: [] for view in views}
        labels = []
        label_info = []
        meta = []
        chunk_info = []
        has_video = False
        has_kpt = False

        for sample in batch:
            sample_labels = sample.get("label")
            if sample_labels is None:
                continue
            if sample_labels.ndim == 0:
                sample_labels = sample_labels.view(1)
            labels.append(sample_labels)

            sample_label_info = sample.get("label_info", [])
            if sample_label_info:
                if isinstance(sample_label_info, list):
                    label_info.extend(sample_label_info)
                else:
                    label_info.append(sample_label_info)

            seg_count = int(sample_labels.shape[0])
            sample_meta = sample.get("meta")
            if sample_meta is not None:
                for seg_idx in range(seg_count):
                    meta_entry = dict(sample_meta)
                    if sample_labels.ndim > 0 and seg_count > 1:
                        meta_entry["segment_idx"] = seg_idx
                        meta_entry["segment_count"] = seg_count
                    meta.append(meta_entry)
                    if sample_meta.get("chunk_info") is not None:
                        chunk_entry = dict(sample_meta["chunk_info"])
                        if sample_labels.ndim > 0 and seg_count > 1:
                            chunk_entry["segment_idx"] = seg_idx
                            chunk_entry["segment_count"] = seg_count
                        chunk_info.append(chunk_entry)

            sample_videos = sample.get("video")
            if isinstance(sample_videos, dict):
                for view in views:
                    if sample_videos.get(view) is not None:
                        video_tensor = sample_videos[view]
                        if video_tensor.ndim == 4:
                            video_tensor = video_tensor.unsqueeze(0)
                        video_lists[view].append(video_tensor)
                        has_video = True

            sample_kpts = sample.get("sam3d_kpt")
            if isinstance(sample_kpts, dict):
                for view in views:
                    if sample_kpts.get(view) is not None:
                        kpt_tensor = sample_kpts[view]
                        if kpt_tensor.ndim == 3:
                            kpt_tensor = kpt_tensor.unsqueeze(0)
                        kpt_lists[view].append(kpt_tensor)
                        has_kpt = True

        label_tensor = (
            torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long)
        )

        video_out = None
        if has_video:
            video_out = {
                view: (
                    torch.cat(video_lists[view], dim=0) if video_lists[view] else None
                )
                for view in views
            }

        kpt_out = None
        if has_kpt:
            kpt_out = {
                view: (
                    torch.cat(kpt_lists[view], dim=0) if kpt_lists[view] else None
                )
                for view in views
            }

        return {
            "video": video_out,
            "sam3d_kpt": kpt_out,
            "label": label_tensor,
            "label_info": label_info,
            "meta": meta,
            "chunk_info": chunk_info,
        }

    def _worker_kwargs(self, num_workers: Optional[int] = None) -> Dict[str, Any]:
        worker_count = self._num_workers if num_workers is None else int(num_workers)
        if worker_count <= 0:
            return {}
        return {
            "persistent_workers": True,
            "prefetch_factor": self._prefetch_factor,
        }

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        sampler = None
        shuffle = True
        if self.batch_unit == "segment" and self.segment_grouped_shuffle:
            sampler = SegmentGroupedSampler(
                self.train_gait_dataset,
                seed=self.segment_shuffle_seed,
                shuffle=True,
            )
            shuffle = False

        train_data_loader = DataLoader(
            self.train_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            collate_fn=self._collate_fn,
            **self._worker_kwargs(self._num_workers),
        )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        val_data_loader = DataLoader(
            self.val_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._val_num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=self._collate_fn,
            **self._worker_kwargs(self._val_num_workers),
        )

        return val_data_loader

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        test_data_loader = DataLoader(
            self.test_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._test_num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=self._collate_fn,
            **self._worker_kwargs(self._test_num_workers),
        )

        return test_data_loader
