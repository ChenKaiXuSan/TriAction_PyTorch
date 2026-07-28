#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/cross_validation.py
Project: /workspace/skeleton/project
Created Date: Friday March 22nd 2024
Author: Kaixu Chen
-----
Comment:
This module defines a single train/validation split.
根据label文件夹中的标注文件，配对对应的视频文件，构建样本列表。
划分结果会被保存到指定的index_mapping目录下的index.json文件中，以便后续加载使用。
不实用person22，23的数据。

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from project.map_config import (
    ENV_KEY_TO_FOLDER,
    CAM_NAMES,
    VideoSample,
)

logger = logging.getLogger(__name__)


class DefineCrossValidation(object):
    """
    New behavior:
      - build samples from:
          videos/{person}/{env_folder}/{cam}.mp4
          label/person_{person}_{day|night}_{high|low}_h265.json
      - produce one train/val split
      - no sampler or K-fold loop
    """

    def __init__(self, config) -> None:
        self.video_path: Path = Path(
            config.paths.video_path
        )  # e.g. /workspace/data/videos
        self.annotation_path: Path = Path(
            config.paths.annotation_path
        )  # e.g. /workspace/data/label
        self.sam3d_results_path: Path = Path(
            config.paths.sam3d_results_path
        )  # e.g. /workspace/data/sam3d_body_results_right

        self.index_mapping: Path = Path(
            config.paths.index_mapping
        )  # folder to save/load index json

        # Magic move configuration
        self.enable_magic_move: bool = bool(getattr(config.data, "magic_move", False))
        self.magic_move_ratio: float = float(
            getattr(config.data, "magic_move_ratio", 0.1)
        )
        self.magic_move_seed: int = int(getattr(config.data, "magic_move_seed", 0))

        # Split mode. "magic_move" moves a random ratio of videos to val and
        # leaves the same person on both sides; "person_kfold" holds out whole
        # persons so val measures generalisation to unseen drivers.
        self.split_mode: str = str(getattr(config.data, "split_mode", "magic_move"))
        if self.split_mode not in {"magic_move", "person_kfold"}:
            raise ValueError(
                f"Unsupported data.split_mode={self.split_mode!r}; "
                "expected 'magic_move' or 'person_kfold'."
            )
        self.num_folds: int = int(getattr(config.data, "num_folds", 5))
        self.fold: int = int(getattr(config.data, "fold", 0))
        self.fold_seed: int = int(getattr(config.data, "fold_seed", 42))
        # nested: the held-out fold becomes test, and a slice of the remaining
        # persons becomes val, so checkpoints are never selected on the data
        # they are reported on
        self.nested_val: bool = bool(getattr(config.data, "nested_val", False))
        self.nested_val_persons: int = int(getattr(config.data, "nested_val_persons", 3))
        if self.split_mode == "person_kfold":
            if self.num_folds < 2:
                raise ValueError(f"data.num_folds must be >= 2, got {self.num_folds}")
            if not 0 <= self.fold < self.num_folds:
                raise ValueError(
                    f"data.fold must be in [0, {self.num_folds}), got {self.fold}"
                )

    # --------- helpers ---------
    @staticmethod
    def _parse_label_filename(p: Path) -> Tuple[str, str, str]:
        """
        person_01_night_high_h265.json -> ("01", "night", "high")
        """
        stem = p.stem  # person_01_night_high_h265
        parts = stem.split("_")
        # 最少应满足：person, 01, night, high, h265
        if len(parts) < 5 or parts[0] != "person":
            raise ValueError(f"Unexpected label filename: {p.name}")
        person_id = parts[1]
        daynight = parts[2]
        highlow = parts[3]
        return person_id, daynight, highlow

    def _collect_one_sample(self, label_path: Path) -> VideoSample | None:
        person_id, daynight, highlow = self._parse_label_filename(label_path)

        # label中的环境 -> 视频文件夹中文名
        if (daynight, highlow) not in ENV_KEY_TO_FOLDER:
            # 不认识的命名就跳过
            return None

        env_folder = ENV_KEY_TO_FOLDER[(daynight, highlow)]
        env_key = f"{daynight}_{highlow}"

        # video root: videos/01/夜多/
        vid_dir = self.video_path / person_id / env_folder
        if not vid_dir.exists():
            # 你的数据可能是 videos/01/... 但 label 是 person_01...
            # 如果视频路径是 01 而 person_id 是 "01" 这没问题；
            # 如果是 "1" vs "01" 才会找不到，需要你统一命名
            return None

        videos: Dict[str, Path] = {}
        for cam in CAM_NAMES:
            mp4 = vid_dir / f"{cam}.mp4"
            if mp4.exists():
                videos[cam] = mp4

        # 至少要有一个视频才算 sample
        if len(videos) == 0:
            return None

        # * Collect SAM 3D body keypoints directory paths (optional)
        # sam3d_results_path/person_id/env_folder/cam/
        sam3d_kpts: Dict[str, Path] = {}
        for cam in CAM_NAMES:
            kpt_dir = self.sam3d_results_path / person_id / env_folder / cam
            if kpt_dir.exists():
                sam3d_kpts[cam] = kpt_dir

        return VideoSample(
            person_id=person_id,
            env_folder=env_folder,
            env_key=env_key,
            label_path=label_path,
            videos=videos,
            sam3d_kpts=sam3d_kpts if len(sam3d_kpts) > 0 else None,
        )

    def build_samples(self) -> List[VideoSample]:
        """
        Scan label directory, pair videos, return samples list.
        """
        label_files = sorted(self.annotation_path.glob("person_*_*.json"))
        samples: List[VideoSample] = []
        for lp in label_files:
            try:
                s = self._collect_one_sample(lp)
            except Exception:
                s = None
            if s is not None:
                samples.append(s)
        return samples

    @staticmethod
    def build_single_split(samples: List[VideoSample]) -> Dict[str, List[VideoSample]]:
        """Use all collected samples as the initial single training split."""
        return {"train": samples, "val": []}

    @staticmethod
    def person_kfold_split(
        samples: List[VideoSample],
        num_folds: int,
        fold: int,
        seed: int = 42,
        nested_val_persons: int = 0,
    ) -> Dict[str, List[VideoSample]]:
        """Hold out every video of one person group as validation.

        Persons -- not videos -- are the unit: with a per-video split the same
        driver appears in train and val, so the score measures memorising
        people rather than generalising to new ones.
        """
        persons = sorted({s.person_id for s in samples})
        if num_folds > len(persons):
            raise ValueError(
                f"num_folds={num_folds} exceeds the {len(persons)} available persons."
            )
        shuffled = list(persons)
        random.Random(seed).shuffle(shuffled)
        # stride assignment keeps the folds within one person of each other
        held_out = set(shuffled[fold::num_folds])
        split = {
            "train": [s for s in samples if s.person_id not in held_out],
            "val": [s for s in samples if s.person_id in held_out],
        }
        if not nested_val_persons:
            return split

        # carve an inner validation set out of the training persons; the
        # held-out fold is kept purely for reporting
        inner_pool = [p for p in shuffled if p not in held_out]
        if nested_val_persons >= len(inner_pool):
            raise ValueError(
                f"nested_val_persons={nested_val_persons} leaves no training persons "
                f"(only {len(inner_pool)} available outside the fold)."
            )
        inner_val = set(inner_pool[:nested_val_persons])
        return {
            "train": [s for s in split["train"] if s.person_id not in inner_val],
            "val": [s for s in split["train"] if s.person_id in inner_val],
            "test": split["val"],
        }

    def magic_move(
        self,
        dataset_split: Dict[str, List[VideoSample]],
        ratio: float = 0.1,
        seed: int = 0,
    ) -> Dict[str, List[VideoSample]]:
        """
        Move a portion of train samples into validation for the single split.
        """
        if ratio <= 0:
            return dataset_split

        rng = random.Random(seed)
        train_samples = list(dataset_split.get("train", []))
        val_samples = list(dataset_split.get("val", []))

        if len(train_samples) == 0:
            return {"train": train_samples, "val": val_samples}

        move_count = int(len(train_samples) * ratio)
        if move_count <= 0 and len(train_samples) > 1:
            move_count = 1

        rng.shuffle(train_samples)
        moved = train_samples[:move_count]
        remaining = train_samples[move_count:]

        return {
            "train": remaining,
            "val": val_samples + moved,
        }

    # --------- main entry ---------
    def prepare(self):
        samples = self.build_samples()
        if len(samples) == 0:
            raise RuntimeError(
                f"No valid samples found. Please check:\n"
                f"  video_path={self.video_path}\n"
                f"  annotation_path={self.annotation_path}\n"
                f"  label filename format: person_XX_(day|night)_(high|low)_h265.json\n"
                f"  video structure: videos/XX/(夜多|夜少|昼多|昼少)/(front|right|left).mp4"
            )

        if self.split_mode == "person_kfold":
            split = self.person_kfold_split(
                samples,
                self.num_folds,
                self.fold,
                self.fold_seed,
                nested_val_persons=self.nested_val_persons if self.nested_val else 0,
            )
            logger.info(
                "person_kfold fold %d/%d: %d train videos, %d val videos "
                "(val persons: %s)",
                self.fold,
                self.num_folds,
                len(split["train"]),
                len(split["val"]),
                sorted({s.person_id for s in split["val"]}),
            )
            return split

        return self.build_single_split(samples)

    @staticmethod
    def _serialize_sample(sample: VideoSample) -> Dict[str, Any]:
        return {
            "person_id": sample.person_id,
            "env_folder": sample.env_folder,
            "env_key": sample.env_key,
            "label_path": str(sample.label_path),
            "videos": {k: str(v) for k, v in sample.videos.items()},
            "sam3d_kpts": {k: str(v) for k, v in sample.sam3d_kpts.items()}
            if sample.sam3d_kpts
            else None,
        }

    @staticmethod
    def _deserialize_sample(item: Dict[str, Any]) -> VideoSample:
        sam3d_kpts = (
            {kk: Path(vv) for kk, vv in item["sam3d_kpts"].items()}
            if item.get("sam3d_kpts")
            else None
        )
        return VideoSample(
            person_id=item["person_id"],
            env_folder=item["env_folder"],
            env_key=item["env_key"],
            label_path=Path(item["label_path"]),
            videos={kk: Path(vv) for kk, vv in item["videos"].items()},
            sam3d_kpts=sam3d_kpts,
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Save/load fold index json
        """
        target_dir = self.index_mapping
        target_dir.mkdir(parents=True, exist_ok=True)

        # Use config values, with kwargs override
        enable_magic_move = self.enable_magic_move
        magic_move_ratio = self.magic_move_ratio
        magic_move_seed = self.magic_move_seed

        if self.split_mode == "person_kfold":
            nested_tag = f"_nested{self.nested_val_persons}" if self.nested_val else ""
            index_name = (
                f"index_person_fold{self.fold}of{self.num_folds}"
                f"_seed{self.fold_seed}{nested_tag}.json"
            )
            enable_magic_move = False  # folds are already disjoint by person
        else:
            index_name = (
                "index_single_magicmove.json"
                if enable_magic_move
                else "index_single.json"
            )
        index_file = target_dir / index_name

        if not index_file.exists():
            dataset_split = self.prepare()
            if enable_magic_move:
                dataset_split = self.magic_move(
                    dataset_split, ratio=magic_move_ratio, seed=magic_move_seed
                )

            # serialize
            serial: Dict[str, Any] = {
                key: [self._serialize_sample(s) for s in dataset_split[key]]
                for key in ("train", "val", "test")
                if key in dataset_split
            }

            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(serial, f, ensure_ascii=False, indent=2)

            return dataset_split

        # load
        with open(index_file, "r", encoding="utf-8") as f:
            serial = json.load(f)

        if "train" not in serial or "val" not in serial:
            raise ValueError(
                f"{index_file} is not a single-split index. "
                "Delete it and regenerate the dataset index."
            )

        return {
            key: [self._deserialize_sample(item) for item in serial[key]]
            for key in ("train", "val", "test")
            if key in serial
        }
