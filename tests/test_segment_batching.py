from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from project.dataloader.data_loader import DriverDataModule, SegmentGroupedSampler
from project.dataloader.whole_video_dataset import LabeledVideoDataset
from project.map_config import VideoSample


class FakeSegmentDataset(LabeledVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_calls = []

    def _get_fps_cached(self, path):
        return 1

    def _get_label_dict_by_annotator_cached(self, label_path):
        return {
            "1": {
                "timeline_list": [
                    {"start": 0, "end": 3, "label": "left"},
                    {"start": 5, "end": 8, "label": "right"},
                ]
            }
        }

    def _load_one_view(self, path, start_sec=None, end_sec=None):
        self.load_calls.append((str(path), start_sec, end_sec))
        start = int(start_sec or 0)
        end = int(end_sec if end_sec is not None else 10)
        frames = torch.arange(10 * 3 * 4 * 4, dtype=torch.float32).reshape(10, 3, 4, 4)
        return frames[start:end], 1


class SegmentBatchingTests(unittest.TestCase):
    def _sample(self):
        return VideoSample(
            person_id="01",
            env_folder="昼多い",
            env_key="day_high",
            label_path=Path("person_01_day_high_h265.json"),
            videos={
                "front": Path("front.mp4"),
                "left": Path("left.mp4"),
                "right": Path("right.mp4"),
            },
            sam3d_kpts=None,
        )

    def _sample_with_kpts(self, kpt_dir: Path):
        sample = self._sample()
        return VideoSample(
            person_id=sample.person_id,
            env_folder=sample.env_folder,
            env_key=sample.env_key,
            label_path=sample.label_path,
            videos=sample.videos,
            sam3d_kpts={"front": kpt_dir},
        )

    def test_segment_mode_indexes_individual_labeled_segments(self):
        dataset = FakeSegmentDataset(
            experiment="test",
            index_mapping=[self._sample()],
            annotation_dict={"01": {"昼多い": {"start": 0, "end": 10}}},
            transform=lambda frames: frames[:2],
            max_video_frames=10,
            view_name=["front"],
            batch_unit="segment",
        )

        self.assertEqual(len(dataset), 4)
        sample = dataset[0]

        self.assertEqual(sample["label"].item(), 0)
        self.assertEqual(sample["label_info"], "left")
        self.assertEqual(tuple(sample["video"]["front"].shape), (3, 2, 4, 4))
        self.assertEqual(sample["meta"]["segment_idx"], 0)
        self.assertEqual(sample["meta"]["segment_count"], 4)

    def test_collate_stacks_segment_items_into_true_batch(self):
        dataset = FakeSegmentDataset(
            experiment="test",
            index_mapping=[self._sample()],
            annotation_dict={"01": {"昼多い": {"start": 0, "end": 10}}},
            transform=lambda frames: frames[:2],
            max_video_frames=10,
            view_name=["front"],
            batch_unit="segment",
        )
        data_module = DriverDataModule.__new__(DriverDataModule)
        data_module.view_name = ["front"]
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=DriverDataModule._collate_fn.__get__(data_module, DriverDataModule),
        )
        batch = next(iter(loader))

        self.assertEqual(tuple(batch["label"].shape), (2,))
        self.assertEqual(tuple(batch["video"]["front"].shape), (2, 3, 2, 4, 4))
        self.assertEqual(len(batch["meta"]), 2)
        self.assertEqual([m["segment_idx"] for m in batch["meta"]], [0, 1])
        self.assertEqual([m["segment_count"] for m in batch["meta"]], [4, 4])
        self.assertEqual([c["segment_idx"] for c in batch["chunk_info"]], [0, 1])

    def test_segment_mode_loads_chunk_range_not_segment_range(self):
        dataset = FakeSegmentDataset(
            experiment="test",
            index_mapping=[self._sample()],
            annotation_dict={"01": {"昼多い": {"start": 0, "end": 10}}},
            transform=lambda frames: frames[:2],
            max_video_frames=10,
            view_name=["front"],
            batch_unit="segment",
        )

        _ = dataset[1]

        self.assertEqual(dataset.load_calls, [("front.mp4", 0.0, 10.0)])

    def test_grouped_sampler_keeps_segments_from_same_source_adjacent(self):
        dataset = FakeSegmentDataset(
            experiment="test",
            index_mapping=[self._sample(), self._sample()],
            annotation_dict={"01": {"昼多い": {"start": 0, "end": 10}}},
            transform=lambda frames: frames[:2],
            max_video_frames=10,
            view_name=["front"],
            batch_unit="segment",
        )
        sampler = SegmentGroupedSampler(dataset, seed=0, shuffle=False)

        indices = list(sampler)
        source_indices = [dataset._segment_index[i]["source_index"] for i in indices]

        self.assertEqual(source_indices, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_sam3d_loader_uses_absolute_frame_filenames_and_output_payload(self):
        dataset = FakeSegmentDataset(
            experiment="test",
            index_mapping=[self._sample()],
            annotation_dict={"01": {"昼多い": {"start": 0, "end": 10}}},
            transform=lambda frames: frames[:2],
            max_video_frames=10,
            view_name=["front"],
            batch_unit="segment",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            kpt_dir = Path(tmp_dir)
            for frame_idx in range(150, 155):
                keypoints = np.full((2, 3), frame_idx, dtype=np.float32)
                np.savez(
                    kpt_dir / f"{frame_idx:06d}_sam3d_body.npz",
                    output=np.array({"pred_keypoints_3d": keypoints}, dtype=object),
                )

            kpts = dataset._load_sam3d_body_kpts(kpt_dir, start_frame=152, end_frame=154)

        self.assertIsNotNone(kpts)
        self.assertEqual(tuple(kpts.shape), (2, 2, 3))
        self.assertEqual(kpts[:, 0, 0].tolist(), [152.0, 153.0])

    def test_segment_mode_filters_segments_without_requested_keypoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            kpt_dir = Path(tmp_dir)
            for frame_idx in range(5, 8):
                keypoints = np.full((2, 3), frame_idx, dtype=np.float32)
                np.savez(
                    kpt_dir / f"{frame_idx:06d}_sam3d_body.npz",
                    output=np.array({"pred_keypoints_3d": keypoints}, dtype=object),
                )

            dataset = FakeSegmentDataset(
                experiment="test",
                index_mapping=[self._sample_with_kpts(kpt_dir)],
                annotation_dict={"01": {"昼多い": {"start": 0, "end": 10}}},
                transform=lambda frames: frames[:2],
                max_video_frames=10,
                view_name=["front"],
                batch_unit="segment",
                load_kpt=True,
                kpt_temporal_subsample_num=2,
            )

            self.assertEqual(len(dataset), 1)
            sample = dataset[0]

        self.assertEqual(sample["label_info"], "right")
        self.assertEqual(tuple(sample["sam3d_kpt"]["front"].shape), (2, 2, 3))


if __name__ == "__main__":
    unittest.main()
