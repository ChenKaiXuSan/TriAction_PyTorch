import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.dataloader.whole_video_dataset import LabeledVideoDataset


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


if __name__ == "__main__":
    unittest.main()
