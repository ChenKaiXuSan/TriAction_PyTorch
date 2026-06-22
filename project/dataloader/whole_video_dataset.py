#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List, Tuple
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

from project.dataloader.prepare_label_dict import prepare_label_dict
from project.map_config import (
    VideoSample,
    label_mapping_Dict,
    normalize_label_to_4_class,
)

logger = logging.getLogger(__name__)


class LabeledVideoDataset(Dataset):
    """
    Multi-view labeled video dataset.

    Output:
        sample["video"][view] : Tensor (B, T, C, H, W)  # segments split by label timeline
        sample["label"]       : LongTensor (B,)
        sample["label_info"]  : List[str]
        sample["meta"]        : dict
    """

    def __init__(
        self,
        experiment: str,
        index_mapping: List[VideoSample],
        annotation_dict: Dict[str, Any],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decode_audio: bool = False,
        load_rgb: bool = True,
        load_kpt: bool = False,
        max_video_frames: Optional[int] = None,  # 如果设置，将长video分块加载
        view_name: list = ["front", "left", "right"],
        annotator_id: Optional[int] = None,
        kpt_temporal_subsample_num: int = 8,
        batch_unit: str = "chunk",
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._index_mapping = index_mapping
        self._annotation_dict = annotation_dict
        self._transform = transform
        self._decode_audio = bool(decode_audio)
        self.load_rgb = bool(load_rgb)
        self.load_kpt = bool(load_kpt)
        self._annotator_id = annotator_id
        self.kpt_temporal_subsample_num = int(kpt_temporal_subsample_num)
        self.batch_unit = str(batch_unit)
        if self.batch_unit not in {"chunk", "segment"}:
            raise ValueError(
                f"Unsupported batch_unit={self.batch_unit}. "
                "Expected 'chunk' or 'segment'."
            )

        self.view_name = view_name

        # Video chunking to avoid OOM during loading
        self.max_video_frames = max_video_frames
        self._chunked_index: List[Dict[str, Any]] = []
        self._segment_index: List[Dict[str, Any]] = []

        # label mapping: {class_id: "label_name"} -> {"label_name": class_id}
        self._label_to_id: Dict[str, int] = {
            v: int(k) for k, v in label_mapping_Dict.items()
        }

        # ===== Performance Optimization: Caching =====
        # FPS cache: avoid repeated fps probing
        self._fps_cache: Dict[str, int] = {}

        # LRU frame cache: store recently loaded frames
        # Key: (video_path, start_sec, end_sec)
        self._frame_cache: OrderedDict[
            Tuple[str, Optional[float], Optional[float]], torch.Tensor
        ] = OrderedDict()
        self._cache_max_size = 2  # Keep most recent 2 videos in memory
        self._cache_memory_limit_mb = 4096  # ~4GB max cache

        # Label cache + valid index mapping (for proper shuffle with unlabeled skip)
        self._label_cache: Dict[str, Dict[str, Any]] = {}
        self._valid_source_indices: List[int] = []

        # Build chunked index if max_video_frames is set
        if self.max_video_frames is not None:
            self._build_chunked_index()
            logger.info(
                f"Video chunking enabled: {len(self._index_mapping)} videos -> "
                f"{len(self._chunked_index)} chunks (max {self.max_video_frames} frames/chunk)"
            )

        if self.batch_unit == "segment":
            self._build_segment_index()
            logger.info(
                f"Segment batching enabled: {len(self._segment_index)} labeled segments"
            )
        else:
            self._build_valid_source_indices()
            source_total = (
                len(self._chunked_index)
                if self.max_video_frames is not None
                else len(self._index_mapping)
            )
            logger.info(
                f"Labeled sample filtering: kept {len(self._valid_source_indices)}/{source_total} samples"
            )

    def _read_video_with_retry(
        self,
        path: str,
        kwargs: Dict[str, Any],
        attempts: int = 3,
        delay_sec: float = 0.2,
    ):
        """Retry transient PyAV/ffmpeg EAGAIN failures from concurrent decoding."""
        last_error = None
        for attempt in range(1, int(attempts) + 1):
            try:
                return read_video(path, **kwargs)
            except BlockingIOError as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                logger.warning(
                    "read_video transient BlockingIOError for %s; retry %s/%s",
                    path,
                    attempt,
                    attempts,
                )
                time.sleep(float(delay_sec) * attempt)
            except OSError as exc:
                if getattr(exc, "errno", None) != 11:
                    raise
                last_error = exc
                if attempt >= attempts:
                    break
                logger.warning(
                    "read_video transient errno=11 for %s; retry %s/%s",
                    path,
                    attempt,
                    attempts,
                )
                time.sleep(float(delay_sec) * attempt)
        raise last_error

    def _get_annotation_range(
        self, item: VideoSample
    ) -> Tuple[int, Optional[int]]:
        person_key = item.person_id
        env_folder = item.env_folder
        anno_start_frame = 0
        anno_end_frame: Optional[int] = None

        if (
            person_key in self._annotation_dict
            and env_folder in self._annotation_dict[person_key]
        ):
            frame_info = self._annotation_dict[person_key][env_folder]
            anno_start_frame = int(frame_info.get("start", 0))
            end_value = frame_info.get("end")
            if end_value is not None:
                anno_end_frame = int(end_value)

        return max(0, anno_start_frame), anno_end_frame

    def _get_label_dict_by_annotator_cached(self, label_path: Path) -> Dict[str, Any]:
        path_str = str(label_path)
        if path_str not in self._label_cache:
            self._label_cache[path_str] = prepare_label_dict(
                label_path,
                annotator=self._annotator_id,
            )
        return self._label_cache[path_str]

    def _get_timeline_in_abs_range(
        self,
        item: VideoSample,
        abs_start: int,
        abs_end: Optional[int],
    ) -> List[Dict[str, Any]]:
        label_dict_by_annotator = self._get_label_dict_by_annotator_cached(item.label_path)

        selected_annotator_key: Optional[str] = None
        if self._annotator_id is not None:
            selected_annotator_key = str(self._annotator_id)
            if selected_annotator_key not in label_dict_by_annotator:
                return []

        if selected_annotator_key is None and len(label_dict_by_annotator) > 0:
            selected_annotator_key = sorted(label_dict_by_annotator.keys())[0]

        if selected_annotator_key is None:
            return []

        timeline_list = label_dict_by_annotator[selected_annotator_key].get(
            "timeline_list", []
        )

        filtered: List[Dict[str, Any]] = []
        range_end = abs_end if abs_end is not None else float("inf")
        for seg in timeline_list:
            seg_start = int(seg["start"])
            seg_end = int(seg["end"])
            if seg_end <= abs_start or seg_start >= range_end:
                continue
            clipped_start = max(abs_start, seg_start)
            clipped_end = min(range_end, seg_end)
            if clipped_end <= clipped_start:
                continue
            filtered.append(
                {
                    "start": clipped_start,
                    "end": clipped_end,
                    "label": seg["label"],
                }
            )

        return filtered

    def _build_valid_source_indices(self) -> None:
        self._valid_source_indices = []

        if self.max_video_frames is not None:
            for source_index, chunk_info in enumerate(self._chunked_index):
                chunk_start_frame = int(chunk_info["chunk_start_frame"])
                chunk_end_frame = chunk_info["chunk_end_frame"]
                if chunk_end_frame is None or int(chunk_end_frame) > chunk_start_frame:
                    self._valid_source_indices.append(source_index)
        else:
            for source_index, item in enumerate(self._index_mapping):
                anno_start_frame, anno_end_frame = self._get_annotation_range(item)
                if anno_end_frame is None or anno_end_frame > anno_start_frame:
                    self._valid_source_indices.append(source_index)

    def _build_chunked_index(self) -> None:
        """
        将长video分成多个chunks，每个chunk最多包含max_video_frames帧。
        这样可以避免加载超长video时OOM。

        Creates a new index where each item represents a chunk:
        {
            'original_item': VideoSample,
            'chunk_start_frame': int,
            'chunk_end_frame': int,
            'chunk_idx': int,
            'total_chunks': int,
        }
        """
        for item in self._index_mapping:
            # Get video total frames from annotation
            person_key = item.person_id
            env_folder = item.env_folder

            total_frames = 0
            start_frame_offset = 0
            end_frame = 0

            if (
                person_key in self._annotation_dict
                and env_folder in self._annotation_dict[person_key]
            ):
                frame_info = self._annotation_dict[person_key][env_folder]
                start_frame_offset = int(frame_info.get("start", 0))
                end_frame = int(frame_info.get("end", 0))

            start_frame_offset = max(0, start_frame_offset)
            end_frame = max(start_frame_offset, end_frame)
            total_frames = end_frame - start_frame_offset

            # 如果无法获取帧数或帧数为0，跳过分块，创建单个item
            if total_frames <= 0:
                self._chunked_index.append(
                    {
                        "original_item": item,
                        "chunk_start_frame": 0,
                        "chunk_end_frame": None,  # Load all
                        "chunk_idx": 0,
                        "total_chunks": 1,
                        "start_frame_offset": 0,
                    }
                )
                continue

            # Calculate number of chunks needed
            num_chunks = (
                total_frames + self.max_video_frames - 1
            ) // self.max_video_frames

            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * self.max_video_frames
                chunk_end = min(chunk_start + self.max_video_frames, total_frames)

                self._chunked_index.append(
                    {
                        "original_item": item,
                        "chunk_start_frame": chunk_start,
                        "chunk_end_frame": chunk_end,
                        "chunk_idx": chunk_idx,
                        "total_chunks": num_chunks,
                        "start_frame_offset": start_frame_offset,
                    }
                )

    def __len__(self) -> int:
        if self.batch_unit == "segment":
            return len(self._segment_index)
        return len(self._valid_source_indices)

    def _source_context(self, source_index: int) -> Dict[str, Any]:
        if self.max_video_frames is not None:
            chunk_info = self._chunked_index[source_index]
            item = chunk_info["original_item"]
            chunk_start_frame = int(chunk_info["chunk_start_frame"])
            chunk_end_frame = chunk_info["chunk_end_frame"]
            start_frame_offset = int(chunk_info["start_frame_offset"])
            loaded_abs_start = start_frame_offset + chunk_start_frame
            loaded_abs_end = start_frame_offset + (
                int(chunk_end_frame)
                if chunk_end_frame is not None
                else chunk_start_frame + int(self.max_video_frames)
            )
            return {
                "item": item,
                "chunk_start_frame": chunk_start_frame,
                "chunk_end_frame": chunk_end_frame,
                "chunk_idx": int(chunk_info["chunk_idx"]),
                "total_chunks": int(chunk_info["total_chunks"]),
                "start_frame_offset": start_frame_offset,
                "loaded_abs_start": loaded_abs_start,
                "loaded_abs_end": loaded_abs_end,
            }

        item = self._index_mapping[source_index]
        anno_start_frame, anno_end_frame = self._get_annotation_range(item)
        loaded_abs_start = anno_start_frame
        loaded_abs_end = anno_end_frame
        return {
            "item": item,
            "chunk_start_frame": 0,
            "chunk_end_frame": None,
            "chunk_idx": 0,
            "total_chunks": 1,
            "start_frame_offset": anno_start_frame,
            "loaded_abs_start": loaded_abs_start,
            "loaded_abs_end": loaded_abs_end,
        }

    def _build_segment_index(self) -> None:
        source_total = (
            len(self._chunked_index)
            if self.max_video_frames is not None
            else len(self._index_mapping)
        )
        for source_index in range(source_total):
            context = self._source_context(source_index)
            item = context["item"]
            loaded_abs_start = int(context["loaded_abs_start"])
            loaded_abs_end = context["loaded_abs_end"]
            if loaded_abs_end is None:
                continue
            loaded_abs_end = int(loaded_abs_end)
            total_frames = max(0, loaded_abs_end - loaded_abs_start)
            if total_frames <= 0:
                continue

            timeline_list = self._get_timeline_in_abs_range(
                item,
                loaded_abs_start,
                loaded_abs_end,
            )
            adjusted_timeline = []
            for seg in timeline_list:
                seg_rel_start = max(0, int(seg["start"]) - loaded_abs_start)
                seg_rel_end = min(total_frames, int(seg["end"]) - loaded_abs_start)
                if seg_rel_end <= seg_rel_start:
                    continue
                adjusted_timeline.append(
                    {
                        "start": seg_rel_start,
                        "end": seg_rel_end,
                        "label": str(seg["label"]),
                    }
                )

            filled_timeline = self._fill_tail_as_front(adjusted_timeline, total_frames)
            valid_segments = []
            for seg in filled_timeline:
                label = normalize_label_to_4_class(str(seg["label"]))
                if label not in self._label_to_id:
                    continue
                valid_segments.append((seg, label))

            segment_count = len(valid_segments)
            for segment_idx, (seg, label) in enumerate(valid_segments):
                self._segment_index.append(
                    {
                        **context,
                        "source_index": source_index,
                        "segment_idx": segment_idx,
                        "segment_count": segment_count,
                        "segment_start_frame": int(seg["start"]),
                        "segment_end_frame": int(seg["end"]),
                        "segment_abs_start": loaded_abs_start + int(seg["start"]),
                        "segment_abs_end": loaded_abs_start + int(seg["end"]),
                        "label": label,
                    }
                )

    # ===== FPS Management =====
    def _get_fps_cached(self, path: Path) -> int:
        """
        Get FPS from cache or probe video metadata.
        Avoids repeated codec initialization.

        Args:
            path: Video file path

        Returns:
            fps: frames per second
        """
        path_str = str(path)
        if path_str not in self._fps_cache:
            # Only probe once per unique video path
            try:
                # Read minimal amount to get metadata
                _, _, info = self._read_video_with_retry(
                    path_str,
                    {
                        "pts_unit": "sec",
                        "output_format": "TCHW",
                        "start_pts": 0.0,
                        "end_pts": 0.001,  # Read first 1ms to get header info
                    },
                )
                fps = int(info.get("video_fps", 0))
                if fps <= 0:
                    raise ValueError(f"Invalid fps={fps} for video: {path}")
                self._fps_cache[path_str] = fps
                logger.debug(f"Cached FPS for {path_str}: {fps}")
            except Exception as e:
                logger.warning(
                    f"Failed to probe fps from {path}: {e}. Will retry on full load."
                )
                # Fall back to full load to get fps
                _, _, info = self._read_video_with_retry(
                    path_str, {"pts_unit": "sec", "output_format": "TCHW"}
                )
                fps = int(info.get("video_fps", 0))
                if fps <= 0:
                    raise ValueError(f"Invalid fps={fps} for video: {path}")
                self._fps_cache[path_str] = fps

        return self._fps_cache[path_str]

    # ---------------- IO ----------------
    def _load_one_view(
        self,
        path: Path,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load one view video and return (video_tchw, fps).
        Uses LRU cache to avoid repeated decoding.

        Args:
            path: Video file path
            start_sec: Start time in seconds (None = from beginning)
            end_sec: End time in seconds (None = to end)

        Returns:
            vframes: (T, C, H, W)
            fps: frames per second
        """
        path_str = str(path)
        cache_key = (path_str, start_sec, end_sec)

        # Check frame cache first
        if cache_key in self._frame_cache:
            logger.debug(f"Frame cache hit: {path_str}[{start_sec}:{end_sec}]")
            # Move to end (most recently used)
            self._frame_cache.move_to_end(cache_key)
            # Still need to return fps from cache
            fps = self._get_fps_cached(path)
            return self._frame_cache[cache_key], fps

        # Actual video loading
        kwargs = {
            "pts_unit": "sec",
            "output_format": "TCHW",
        }

        if start_sec is not None:
            kwargs["start_pts"] = start_sec
        if end_sec is not None:
            kwargs["end_pts"] = end_sec

        vframes, aframes, info = self._read_video_with_retry(str(path), kwargs)
        fps = int(info.get("video_fps", 0))
        if fps <= 0:
            raise ValueError(f"Invalid fps={fps} for video: {path}")

        # Update FPS cache
        self._fps_cache[path_str] = fps

        # Add to frame cache with LRU eviction
        self._frame_cache[cache_key] = vframes
        self._frame_cache.move_to_end(cache_key)  # Mark as most recently used

        # Evict least recently used if cache too large
        while len(self._frame_cache) > self._cache_max_size:
            # Remove oldest
            oldest_key = next(iter(self._frame_cache))
            del self._frame_cache[oldest_key]
            logger.debug(f"LRU evict: {oldest_key[0]}")

        logger.debug(
            f"Cached frame: {path_str}[{start_sec}:{end_sec}] "
            f"cache_size={len(self._frame_cache)}"
        )

        return vframes, fps

    def _load_sam3d_body_kpts(
        self,
        sam3d_dir: Optional[Path],
        start_frame: int,
        end_frame: Optional[int],
    ) -> Optional[torch.Tensor]:
        """Load SAM3D keypoints as ``(T, K, 3)`` for an absolute frame range."""
        if sam3d_dir is None or not Path(sam3d_dir).exists():
            return None

        files = sorted(Path(sam3d_dir).glob("*.npz"))
        if not files:
            return None

        desired_len = None
        if end_frame is not None:
            desired_len = max(0, int(end_frame) - int(start_frame))

        numbered_files = []
        for file_path in files:
            frame_token = file_path.stem.split("_", 1)[0]
            if frame_token.isdigit():
                numbered_files.append((int(frame_token), file_path))

        if numbered_files:
            selected_files = [
                file_path
                for frame_idx, file_path in numbered_files
                if frame_idx >= int(start_frame)
                and (end_frame is None or frame_idx < int(end_frame))
            ]
        elif end_frame is not None and int(end_frame) <= len(files):
            selected_files = files[int(start_frame) : int(end_frame)]
        elif desired_len is not None:
            selected_files = files[:desired_len]
        else:
            selected_files = files[int(start_frame):]

        kpt_frames: List[torch.Tensor] = []
        for npz_path in selected_files:
            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    key = None
                    for candidate in (
                        "keypoints_3d",
                        "pred_keypoints_3d",
                        "pred_joint_coords",
                        "poses",
                        "keypoints",
                        "arr_0",
                    ):
                        if candidate in data:
                            key = candidate
                            break
                    if key is not None:
                        arr = np.asarray(data[key], dtype=np.float32)
                    elif "output" in data:
                        output = data["output"]
                        if isinstance(output, np.ndarray):
                            if output.shape == ():
                                output = output.item()
                            elif output.dtype == object and output.size > 0:
                                output = output.flat[0]
                        if not isinstance(output, dict):
                            continue
                        output_key = None
                        for candidate in (
                            "pred_keypoints_3d",
                            "pred_joint_coords",
                            "keypoints_3d",
                            "keypoints",
                        ):
                            if candidate in output:
                                output_key = candidate
                                break
                        if output_key is None:
                            continue
                        arr = np.asarray(output[output_key], dtype=np.float32)
                    else:
                        continue
            except Exception as exc:
                logger.warning("Failed to load SAM3D keypoints from %s: %s", npz_path, exc)
                continue

            arr = np.squeeze(arr)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim != 2 or arr.shape[-1] < 3:
                continue
            kpt_frames.append(torch.from_numpy(arr[:, :3].astype(np.float32)))

        if not kpt_frames:
            return None
        return torch.stack(kpt_frames, dim=0)

    def _apply_transform(self, video_tchw: torch.Tensor) -> torch.Tensor:
        """
        Apply transform on a segment.

        Expect transform: (T,C,H,W) -> (T,C,H,W) or compatible.
        """
        if self._transform is None:
            return video_tchw
        return self._transform(video_tchw)

    def _apply_kpt_transform(self, kpts_tkc: torch.Tensor) -> torch.Tensor:
        """Uniformly sample keypoints to a fixed temporal length."""
        if kpts_tkc.ndim != 3:
            raise ValueError(f"Expected keypoints (T, K, 3), got {tuple(kpts_tkc.shape)}")
        t = int(kpts_tkc.shape[0])
        if t <= 0:
            raise ValueError("Cannot transform empty keypoint sequence.")
        if self.kpt_temporal_subsample_num <= 0 or t == self.kpt_temporal_subsample_num:
            return kpts_tkc.float()
        indices = torch.linspace(
            0,
            t - 1,
            steps=self.kpt_temporal_subsample_num,
            device=kpts_tkc.device,
        ).long()
        return kpts_tkc.index_select(0, indices).float()

    def _validate_output_shapes(
        self,
        batch_front: Optional[torch.Tensor],
        batch_left: Optional[torch.Tensor],
        batch_right: Optional[torch.Tensor],
        mapped_labels: torch.LongTensor,
        labels: List[str],
        *keypoint_tensors: Optional[torch.Tensor],
    ) -> None:
        """
        Validate output tensor shapes and consistency.

        Args:
            batch_front, batch_left, batch_right: Video tensors (B, C, T, H, W) or None (for backward compat)
            mapped_labels: Label tensor (B,)
            labels: List of label strings
        Raises:
            AssertionError: If shapes are inconsistent
        """
        # Get batch size from available data
        if batch_front is not None:
            B = batch_front.shape[0]
        elif batch_left is not None:
            B = batch_left.shape[0]
        elif batch_right is not None:
            B = batch_right.shape[0]
        else:
            B = mapped_labels.shape[0]

        # Validate batch size consistency
        assert mapped_labels.shape[0] == B, (
            f"Labels batch size {mapped_labels.shape[0]} != expected {B}"
        )
        assert len(labels) == B, f"Labels list length {len(labels)} != expected {B}"

        # Video batch consistency check (all video tensors should have same B if not None)
        video_tensors = [batch_front, batch_left, batch_right]
        valid_video_tensors = [t for t in video_tensors if t is not None]

        if len(valid_video_tensors) > 0:
            for i, tensor in enumerate(valid_video_tensors):
                assert tensor.ndim == 5, (
                    f"Video tensor {i} should be 5D (B, C, T, H, W), got {tensor.ndim}D"
                )
                assert tensor.shape[0] == B, (
                    f"Video tensor {i} batch size {tensor.shape[0]} != expected {B}"
                )
                assert tensor.shape[1] == 3, (
                    f"Video tensor {i} should have 3 channels, got {tensor.shape[1]}"
                )

        for i, tensor in enumerate(t for t in keypoint_tensors if t is not None):
            assert tensor.ndim == 4, (
                f"Keypoint tensor {i} should be 4D (B, T, K, 3), got {tensor.ndim}D"
            )
            assert tensor.shape[0] == B, (
                f"Keypoint tensor {i} batch size {tensor.shape[0]} != expected {B}"
            )
            assert tensor.shape[-1] == 3, (
                f"Keypoint tensor {i} should have XYZ coordinates, got {tensor.shape[-1]}"
            )

    # ---------------- Timeline utils ----------------
    @staticmethod
    def _fill_tail_as_front(
        timeline: List[Dict[str, Any]],
        total_frames: int,
        front_label: str = "front",
    ) -> List[Dict[str, Any]]:
        """
        If the timeline doesn't cover [0, total_frames), fill uncovered gaps as front.
        Assumes timeline items have int-like start/end and end is exclusive.
        """
        if total_frames <= 0:
            return []

        # sort by start
        tl = sorted(
            (
                {
                    "start": int(x["start"]),
                    "end": int(x["end"]),
                    "label": str(x["label"]),
                }
                for x in timeline
                if x is not None and "start" in x and "end" in x and "label" in x
            ),
            key=lambda d: (d["start"], d["end"]),
        )

        filled: List[Dict[str, Any]] = []
        cur = 0

        for seg in tl:
            s, e, lb = seg["start"], seg["end"], seg["label"]
            s = max(0, min(s, total_frames))
            e = max(0, min(e, total_frames))
            if e <= s:
                continue

            # gap before this seg
            if s > cur:
                filled.append({"start": cur, "end": s, "label": front_label})

            filled.append({"start": s, "end": e, "label": lb})
            cur = max(cur, e)

        # tail gap
        if cur < total_frames:
            filled.append({"start": cur, "end": total_frames, "label": front_label})

        return filled

    def split_frame_with_label(
        self,
        front_view: torch.Tensor,  # (T,C,H,W)
        left_view: torch.Tensor,  # (T,C,H,W)
        right_view: torch.Tensor,  # (T,C,H,W)
        timeline_list: List[Dict[str, Any]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[str],
        torch.LongTensor,
    ]:
        """
        Split video frames according to label timeline.

        Returns:
            batch_front: (B, C, T, H, W)
            batch_left: (B, C, T, H, W)
            batch_right: (B, C, T, H, W)
            labels: List[str]
            mapped_labels: (B,)
        """
        assert front_view.shape[0] == left_view.shape[0] == right_view.shape[0], (
            "All views must have the same number of frames"
        )

        T = int(front_view.shape[0])

        # 1) Sort and clean labeled regions. Gaps are filled as front below.
        timeline = sorted(
            (
                {
                    "start": int(x["start"]),
                    "end": int(x["end"]),
                    "label": str(x["label"]),
                }
                for x in timeline_list
                if x is not None and "start" in x and "end" in x and "label" in x
            ),
            key=lambda d: (d["start"], d["end"]),
        )

        batch_front: List[torch.Tensor] = []
        batch_left: List[torch.Tensor] = []
        batch_right: List[torch.Tensor] = []
        labels: List[str] = []
        mapped: List[int] = []

        for seg in timeline:
            s, e, lb = int(seg["start"]), int(seg["end"]), str(seg["label"])
            if e <= s:
                continue

            lb = normalize_label_to_4_class(lb)

            seg_front = self._apply_transform(front_view[s:e])
            seg_left = self._apply_transform(left_view[s:e])
            seg_right = self._apply_transform(right_view[s:e])

            batch_front.append(seg_front)
            batch_left.append(seg_left)
            batch_right.append(seg_right)

            labels.append(lb)
            mapped.append(self._label_to_id.get(lb, -1))  # unknown -> -1

        # Stack video tensors
        batch_front_t = torch.stack(batch_front, dim=0).permute(0, 2, 1, 3, 4)
        batch_left_t = torch.stack(batch_left, dim=0).permute(0, 2, 1, 3, 4)
        batch_right_t = torch.stack(batch_right, dim=0).permute(0, 2, 1, 3, 4)

        mapped_t = torch.tensor(mapped, dtype=torch.long)

        return (
            batch_front_t,
            batch_left_t,
            batch_right_t,
            labels,
            mapped_t,
        )

    def split_kpt_with_label(
        self,
        kpts_by_view: Dict[str, Optional[torch.Tensor]],
        timeline_list: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Optional[torch.Tensor]], List[str], torch.LongTensor]:
        """Split keypoint sequences according to label timeline."""
        timeline = sorted(
            (
                {
                    "start": int(x["start"]),
                    "end": int(x["end"]),
                    "label": str(x["label"]),
                }
                for x in timeline_list
                if x is not None and "start" in x and "end" in x and "label" in x
            ),
            key=lambda d: (d["start"], d["end"]),
        )

        labels: List[str] = []
        mapped: List[int] = []
        per_view_segments: Dict[str, List[torch.Tensor]] = {
            view: [] for view in self.view_name
        }

        for seg in timeline:
            s, e, lb = int(seg["start"]), int(seg["end"]), str(seg["label"])
            if e <= s:
                continue

            lb = normalize_label_to_4_class(lb)
            labels.append(lb)
            mapped.append(self._label_to_id.get(lb, -1))

            for view in self.view_name:
                kpts = kpts_by_view.get(view)
                if kpts is None:
                    continue
                seg_kpts = kpts[s:e]
                if seg_kpts.shape[0] == 0:
                    continue
                per_view_segments[view].append(self._apply_kpt_transform(seg_kpts))

        mapped_t = torch.tensor(mapped, dtype=torch.long)
        batch_kpts: Dict[str, Optional[torch.Tensor]] = {}
        for view, segments in per_view_segments.items():
            batch_kpts[view] = torch.stack(segments, dim=0) if segments else None

        return batch_kpts, labels, mapped_t

    def _getitem_segment(self, index: int) -> Dict[str, Any]:
        segment = self._segment_index[index]
        item = segment["item"]
        fps = self._get_fps_cached(item.videos["front"])
        chunk_abs_start = int(segment["start_frame_offset"]) + int(
            segment["chunk_start_frame"]
        )
        if segment["chunk_end_frame"] is not None:
            chunk_abs_end = int(segment["start_frame_offset"]) + int(
                segment["chunk_end_frame"]
            )
        else:
            chunk_abs_end = int(segment["segment_abs_end"])
        chunk_start_sec = chunk_abs_start / fps
        chunk_end_sec = chunk_abs_end / fps
        segment_rel_start = max(0, int(segment["segment_abs_start"]) - chunk_abs_start)
        segment_rel_end = max(segment_rel_start, int(segment["segment_abs_end"]) - chunk_abs_start)
        requested_views = set(self.view_name)

        video_out: Optional[Dict[str, torch.Tensor]] = None
        if self.load_rgb:
            loaded_views = {}
            for view_name in ["front", "left", "right"]:
                if view_name not in requested_views:
                    continue
                frames, _ = self._load_one_view(
                    item.videos[view_name],
                    chunk_start_sec,
                    chunk_end_sec,
                )
                segment_frames = frames[segment_rel_start:segment_rel_end]
                loaded_views[view_name] = self._apply_transform(segment_frames).permute(
                    1, 0, 2, 3
                )
            video_out = {view: loaded_views[view] for view in self.view_name}

        kpt_out: Optional[Dict[str, torch.Tensor]] = None
        if self.load_kpt:
            kpt_out = {}
            for view_name in self.view_name:
                kpt_dir = (
                    item.sam3d_kpts.get(view_name)
                    if item.sam3d_kpts is not None
                    else None
                )
                kpts = self._load_sam3d_body_kpts(
                    kpt_dir,
                    start_frame=int(segment["segment_abs_start"]),
                    end_frame=int(segment["segment_abs_end"]),
                )
                if kpts is not None:
                    kpt_out[view_name] = self._apply_kpt_transform(kpts)
            if not kpt_out:
                raise RuntimeError(
                    f"No keypoint segment found for source_index={segment['source_index']} "
                    f"(person={item.person_id}, env={item.env_folder})."
                )

        label_name = str(segment["label"])
        mapped_label = torch.tensor(self._label_to_id[label_name], dtype=torch.long)

        return {
            "video": video_out,
            "sam3d_kpt": kpt_out,
            "label": mapped_label,
            "label_info": label_name,
            "meta": {
                "experiment": self._experiment,
                "index": int(segment["source_index"]),
                "person_id": item.person_id,
                "env_folder": item.env_folder,
                "env_key": item.env_key,
                "start_frame": int(segment["segment_start_frame"]),
                "end_frame": int(segment["segment_end_frame"]),
                "fps": fps,
                "is_chunked": self.max_video_frames is not None,
                "segment_idx": int(segment["segment_idx"]),
                "segment_count": int(segment["segment_count"]),
                "chunk_info": {
                    "chunk_idx": int(segment["chunk_idx"]),
                    "total_chunks": int(segment["total_chunks"]),
                    "chunk_start_frame": int(segment["chunk_start_frame"]),
                    "chunk_end_frame": segment["chunk_end_frame"],
                    "absolute_start_frame": int(segment["segment_abs_start"]),
                    "absolute_end_frame": int(segment["segment_abs_end"]),
                    "annotation_start": int(segment["start_frame_offset"]),
                    "segment_idx": int(segment["segment_idx"]),
                    "segment_count": int(segment["segment_count"]),
                }
                if self.max_video_frames is not None
                else None,
            },
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.batch_unit == "segment":
            return self._getitem_segment(index)

        source_index = self._valid_source_indices[index]
        # Handle chunked vs non-chunked index
        if self.max_video_frames is not None:
            chunk_info = self._chunked_index[source_index]
            item = chunk_info["original_item"]
            chunk_start_frame = chunk_info["chunk_start_frame"]
            chunk_end_frame = chunk_info["chunk_end_frame"]
            start_frame_offset = chunk_info["start_frame_offset"]

            # ===== OPTIMIZATION: Use cached FPS instead of probing =====
            fps = self._get_fps_cached(item.videos["front"])

            # Calculate actual start/end in seconds
            # chunk frames are relative to the annotation start
            actual_start_frame = start_frame_offset + chunk_start_frame
            actual_end_frame = start_frame_offset + (
                chunk_end_frame
                if chunk_end_frame
                else chunk_start_frame + self.max_video_frames
            )

            start_sec = actual_start_frame / fps
            end_sec = actual_end_frame / fps

        else:
            item = self._index_mapping[source_index]
            chunk_start_frame = 0
            chunk_end_frame = None
            start_frame_offset = 0

            # Get start/end frame from annotation dict (same as chunked path)
            person_key = item.person_id
            env_folder = item.env_folder
            anno_start_frame = 0
            anno_end_frame = None  # None means load to end

            if (
                person_key in self._annotation_dict
                and env_folder in self._annotation_dict[person_key]
            ):
                frame_info = self._annotation_dict[person_key][env_folder]
                anno_start_frame = int(frame_info.get("start", 0))
                start_frame_offset = max(0, anno_start_frame)
                anno_end_frame = frame_info.get("end")
                if anno_end_frame is not None:
                    anno_end_frame = int(anno_end_frame)

            # ===== OPTIMIZATION: Use cached FPS instead of probing =====
            fps = self._get_fps_cached(item.videos["front"])

            # Convert frame indices to seconds for load_one_view
            start_sec = anno_start_frame / fps if anno_start_frame > 0 else None
            end_sec = anno_end_frame / fps if anno_end_frame is not None else None

            # Metadata: frame bounds (relative to loaded segment)
            start_frame = 0
            end_frame = None  # Will be set to total_frames after loading

        # Load requested RGB views only when needed.
        views = {
            "front": item.videos["front"],
            "left": item.videos["left"],
            "right": item.videos["right"],
        }
        requested_views = set(self.view_name)
        loaded_views: Dict[str, Optional[torch.Tensor]] = {
            "front": None,
            "left": None,
            "right": None,
        }

        loaded_kpts: Dict[str, Optional[torch.Tensor]] = {
            "front": None,
            "left": None,
            "right": None,
        }

        for view_name in ["front", "left", "right"]:
            if self.load_rgb and view_name in requested_views:
                frames, fps_view = self._load_one_view(
                    views[view_name], start_sec, end_sec
                )
                loaded_views[view_name] = frames
                if fps == 0:
                    fps = fps_view

        abs_range_start = (
            start_frame_offset + chunk_start_frame
            if self.max_video_frames is not None
            else start_frame_offset
        )
        abs_range_end = (
            start_frame_offset
            + (
                chunk_end_frame
                if chunk_end_frame is not None
                else chunk_start_frame + self.max_video_frames
            )
            if self.max_video_frames is not None
            else (
                anno_end_frame
                if "anno_end_frame" in locals() and anno_end_frame is not None
                else None
            )
        )

        if self.load_kpt:
            for view_name in ["front", "left", "right"]:
                if view_name not in requested_views:
                    continue
                kpt_dir = (
                    item.sam3d_kpts.get(view_name)
                    if item.sam3d_kpts is not None
                    else None
                )
                loaded_kpts[view_name] = self._load_sam3d_body_kpts(
                    kpt_dir,
                    start_frame=int(abs_range_start),
                    end_frame=int(abs_range_end) if abs_range_end is not None else None,
                )

        ref_frames = None
        for view_name in ["front", "left", "right"]:
            if loaded_views[view_name] is not None:
                ref_frames = loaded_views[view_name]
                break

        ref_kpts = None
        for view_name in ["front", "left", "right"]:
            if loaded_kpts[view_name] is not None:
                ref_kpts = loaded_kpts[view_name]
                break

        if self.load_rgb and ref_frames is None:
            raise ValueError("RGB loading requested but no views were loaded.")
        if self.load_kpt and ref_kpts is None:
            raise ValueError("Keypoint loading requested but no SAM3D keypoints were loaded.")

        front_frames = left_frames = right_frames = None
        if self.load_rgb:
            front_frames = (
                loaded_views["front"]
                if loaded_views["front"] is not None
                else torch.zeros_like(ref_frames)
            )
            left_frames = (
                loaded_views["left"]
                if loaded_views["left"] is not None
                else torch.zeros_like(ref_frames)
            )
            right_frames = (
                loaded_views["right"]
                if loaded_views["right"] is not None
                else torch.zeros_like(ref_frames)
            )

            assert front_frames.shape[0] == left_frames.shape[0] == right_frames.shape[0], (
                "All views must have the same number of frames"
            )
            total_frames = int(front_frames.shape[0])
        elif ref_kpts is not None:
            total_frames = int(ref_kpts.shape[0])
        elif abs_range_end is not None:
            total_frames = max(0, int(abs_range_end) - int(abs_range_start))
        else:
            raise ValueError("Unable to infer sequence length for this sample.")

        if self.load_kpt and ref_kpts is not None:
            for view_name in ["front", "left", "right"]:
                if view_name not in requested_views:
                    continue
                if loaded_kpts[view_name] is None:
                    loaded_kpts[view_name] = torch.zeros_like(ref_kpts)

        # For chunked case, set frame bounds
        if self.max_video_frames is not None:
            start_frame = 0
            end_frame = total_frames
        else:
            # For non-chunked case: end_frame was None, now set it
            end_frame = total_frames

        # ==================== Label Coordinate Conversion ====================
        # 标签处理：不填充 front，直接使用 annotation_dict 中的标注
        #
        # 坐标系统说明：
        # 1. 标签文件中的帧索引是相对于原始完整视频的绝对索引（例如：2000-3000）
        # 2. 加载的视频帧在内存中是从 0 开始的相对索引
        # 3. 需要将标签的绝对索引转换为相对于加载视频的索引
        #
        # 例如：
        #   原始视频: [0 .................. 1000 ........ 2000 ........ 3000 ........ 5000]
        #   标签范围:                              |<--- label [2000, 3000) --->|
        #   加载视频:                              [0 ................... total_frames)
        #   目标索引:                              [0 ................... 1000)
        # ======================================================================

        # Step 1: 确定实际加载的视频在原始视频中的绝对位置
        # Determine the absolute frame range of the loaded video in the original video
        if self.max_video_frames is not None:
            # Chunked mode: 加载了 chunk [chunk_start_frame, chunk_end_frame) 相对于 annotation start
            # 绝对位置 = annotation_start + chunk_position
            loaded_video_abs_start = start_frame_offset + chunk_start_frame
            loaded_video_abs_end = start_frame_offset + (
                chunk_end_frame
                if chunk_end_frame is not None
                else chunk_start_frame + self.max_video_frames
            )
        else:
            # Non-chunked mode: 加载了从 annotation start 到 end 的整段视频
            loaded_video_abs_start = start_frame_offset
            loaded_video_abs_end = start_frame_offset + total_frames

        # Step 2: 加载标签，只获取与当前加载视频重叠的 segments
        # Load labels, filtering to only segments that overlap with loaded video
        # TODO: 下面的操作有问题，感觉会得到空的timeline_list，后续需要修正prepare_label_dict函数
        timeline_list = self._get_timeline_in_abs_range(
            item,
            loaded_video_abs_start,
            loaded_video_abs_end,
        )

        # Step 3: 将标签的绝对帧索引转换为相对于加载视频的索引
        # Convert label's absolute frame indices to relative indices (0-based, relative to loaded video)
        # Note: timeline_list now only contains segments that overlap with loaded video
        logger.debug("[Label Coordinate Conversion]")
        logger.debug(
            f"  Loaded video (absolute): [{loaded_video_abs_start}, {loaded_video_abs_end})"
        )
        logger.debug(f"  Loaded video (relative): [0, {total_frames})")
        logger.debug(f"  Filtered timeline segments: {len(timeline_list)}")

        adjusted_timeline = []
        for seg in timeline_list:
            seg_abs_start = int(seg["start"])  # 标签文件中的绝对起始帧
            seg_abs_end = int(seg["end"])  # 标签文件中的绝对结束帧

            # 转换公式：relative_index = absolute_index - loaded_video_abs_start
            seg_rel_start = seg_abs_start - loaded_video_abs_start
            seg_rel_end = seg_abs_end - loaded_video_abs_start

            # 裁剪到有效范围 [0, total_frames)
            # Clip to valid range [0, total_frames) to handle partial overlaps
            seg_rel_start = max(0, seg_rel_start)
            seg_rel_end = min(total_frames, seg_rel_end)

            if seg_rel_end <= seg_rel_start:
                logger.debug(
                    f"  [{seg_abs_start:5d}, {seg_abs_end:5d}) -> [{seg_rel_start:5d}, {seg_rel_end:5d}) {seg['label']:8s} FILTERED (empty after clipping)"
                )
                continue

            adjusted_timeline.append(
                {"start": seg_rel_start, "end": seg_rel_end, "label": seg["label"]}
            )
            logger.debug(
                f"  [{seg_abs_start:5d}, {seg_abs_end:5d}) -> [{seg_rel_start:5d}, {seg_rel_end:5d}) {seg['label']:8s} INCLUDED"
            )

        timeline_list = self._fill_tail_as_front(adjusted_timeline, total_frames)
        logger.debug(f"  Final timeline segments: {len(timeline_list)}")
        # ==================== End Label Coordinate Conversion ====================

        if len(timeline_list) == 0:
            raise RuntimeError(
                f"No labeled timeline after clipping for source_index={source_index} "
                f"(person={item.person_id}, env={item.env_folder})."
            )

        video_out: Optional[Dict[str, torch.Tensor]] = None
        kpt_out: Optional[Dict[str, torch.Tensor]] = None

        labels: List[str]
        mapped_labels: torch.LongTensor
        batch_front = batch_left = batch_right = None

        if self.load_rgb:
            (
                batch_front,
                batch_left,
                batch_right,
                labels,
                mapped_labels,
            ) = self.split_frame_with_label(
                front_frames,
                left_frames,
                right_frames,
                timeline_list,
            )
            all_video = {
                "front": batch_front,
                "left": batch_left,
                "right": batch_right,
            }
            video_out = {view: all_video[view] for view in self.view_name}

        if self.load_kpt:
            kpt_out, kpt_labels, kpt_mapped_labels = self.split_kpt_with_label(
                loaded_kpts,
                timeline_list,
            )
            kpt_out = {
                view: tensor
                for view, tensor in kpt_out.items()
                if view in self.view_name and tensor is not None
            }
            if not kpt_out:
                raise RuntimeError(
                    f"No keypoint segments found for source_index={source_index} "
                    f"(person={item.person_id}, env={item.env_folder})."
                )
            if not self.load_rgb:
                labels = kpt_labels
                mapped_labels = kpt_mapped_labels

        # Validate output shapes
        self._validate_output_shapes(
            batch_front,
            batch_left,
            batch_right,
            mapped_labels,
            labels,
            *(kpt_out.values() if kpt_out else []),
        )

        return {
            "video": video_out,
            "sam3d_kpt": kpt_out,
            "label": mapped_labels,  # LongTensor (B,)
            "label_info": labels,  # List[str]
            "meta": {
                "experiment": self._experiment,
                "index": source_index,
                "person_id": item.person_id,
                "env_folder": item.env_folder,
                "env_key": item.env_key,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": fps,
                "is_chunked": self.max_video_frames is not None,
                "chunk_info": {
                    "chunk_idx": chunk_info["chunk_idx"],
                    "total_chunks": chunk_info["total_chunks"],
                    "chunk_start_frame": chunk_start_frame,  # Relative to annotation start
                    "chunk_end_frame": chunk_end_frame,  # Relative to annotation start
                    "absolute_start_frame": start_frame_offset
                    + chunk_start_frame,  # Absolute frame index in original video
                    "absolute_end_frame": start_frame_offset
                    + (
                        chunk_end_frame if chunk_end_frame else chunk_start_frame
                    ),  # Absolute frame index in original video
                    "annotation_start": start_frame_offset,  # Annotation start frame
                }
                if self.max_video_frames is not None
                else None,
            },
        }


def whole_video_dataset(
    experiment: str,
    dataset_idx: List[VideoSample],
    annotation_dict: Dict[str, Any],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    load_rgb: bool = True,
    load_kpt: bool = False,
    max_video_frames: Optional[int] = None,
    view_name: List[str] = ["front", "left", "right"],
    annotator_id: Optional[int] = None,
    kpt_temporal_subsample_num: int = 8,
    batch_unit: str = "chunk",
) -> LabeledVideoDataset:
    """
    Create a LabeledVideoDataset for whole video processing.

    Args:
        experiment: Experiment name
        dataset_idx: List of VideoSample items (contains sam3d_kpts paths)
        annotation_dict: Annotation dictionary
        transform: Optional transform to apply to video frames
        max_video_frames: Maximum frames per chunk. If set, long videos will be
            split into multiple chunks to avoid OOM during loading. For example,
            max_video_frames=1000 means videos longer than 1000 frames will be
            split into multiple samples. Recommended: 500-2000 depending on resolution.
            (default: None - load entire video)
        view_name: List of view names to load (default: ["front", "left", "right"])
        annotator_id: Optional annotator id. If provided, only this annotator's
            labels will be used when parsing `item.label_path`.

    Returns:
        LabeledVideoDataset instance
    """
    return LabeledVideoDataset(
        experiment=experiment,
        index_mapping=dataset_idx,
        annotation_dict=annotation_dict,
        transform=transform,
        load_rgb=load_rgb,
        load_kpt=load_kpt,
        max_video_frames=max_video_frames,
        view_name=view_name,
        annotator_id=annotator_id,
        kpt_temporal_subsample_num=kpt_temporal_subsample_num,
        batch_unit=batch_unit,
    )
