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
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from project.map_config import (
    ENV_KEY_TO_FOLDER,
    CAM_NAMES,
    VideoSample,
)


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

        index_name = (
            "index_single_magicmove.json" if enable_magic_move else "index_single.json"
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
                "train": [self._serialize_sample(s) for s in dataset_split["train"]],
                "val": [self._serialize_sample(s) for s in dataset_split["val"]],
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
            "train": [self._deserialize_sample(item) for item in serial["train"]],
            "val": [self._deserialize_sample(item) for item in serial["val"]],
        }
