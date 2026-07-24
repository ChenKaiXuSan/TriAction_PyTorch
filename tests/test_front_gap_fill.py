import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.dataloader.whole_video_dataset import LabeledVideoDataset
from project.map_config import normalize_label_to_4_class


class FrontGapFillTest(unittest.TestCase):
    def test_fill_tail_as_front_fills_all_unlabeled_gaps(self):
        timeline = [
            {"start": 10, "end": 20, "label": "left"},
            {"start": 30, "end": 35, "label": "up"},
        ]

        filled = LabeledVideoDataset._fill_tail_as_front(timeline, total_frames=40)

        self.assertEqual(
            filled,
            [
                {"start": 0, "end": 10, "label": "front"},
                {"start": 10, "end": 20, "label": "left"},
                {"start": 20, "end": 30, "label": "front"},
                {"start": 30, "end": 35, "label": "up"},
                {"start": 35, "end": 40, "label": "front"},
            ],
        )

    def test_timeline_policy_skips_front_gaps_when_front_is_not_a_class(self):
        timeline = [
            {"start": 10, "end": 20, "label": "left_up"},
            {"start": 30, "end": 35, "label": "right_down"},
        ]
        label_to_id = {"left": 0, "right": 1, "down": 2, "up": 3}

        prepared = LabeledVideoDataset._prepare_timeline_for_labels(
            timeline,
            total_frames=40,
            label_to_id=label_to_id,
        )

        self.assertEqual(
            prepared,
            [
                {"start": 10, "end": 20, "label": "up"},
                {"start": 30, "end": 35, "label": "down"},
            ],
        )

    def test_combined_vertical_labels_map_to_vertical_classes(self):
        self.assertEqual(normalize_label_to_4_class("left_up"), "up")
        self.assertEqual(normalize_label_to_4_class("right_up"), "up")
        self.assertEqual(normalize_label_to_4_class("left_down"), "down")
        self.assertEqual(normalize_label_to_4_class("right_down"), "down")


if __name__ == "__main__":
    unittest.main()
