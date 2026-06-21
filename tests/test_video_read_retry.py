import unittest
from unittest.mock import patch

import torch

from project.dataloader.whole_video_dataset import LabeledVideoDataset


class VideoReadRetryTests(unittest.TestCase):
    def test_retries_transient_blocking_io_error(self):
        dataset = LabeledVideoDataset(
            experiment="test",
            index_mapping=[],
            annotation_dict={},
            batch_unit="chunk",
        )
        frames = torch.zeros(1, 3, 4, 4)
        calls = {"count": 0}

        def flaky_read_video(path, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise BlockingIOError(11, "Resource temporarily unavailable")
            return frames, None, {"video_fps": 30}

        with patch(
            "project.dataloader.whole_video_dataset.read_video",
            side_effect=flaky_read_video,
        ):
            loaded_frames, _, info = dataset._read_video_with_retry(
                "video.mp4",
                {"pts_unit": "sec", "output_format": "TCHW"},
                delay_sec=0,
            )

        self.assertIs(loaded_frames, frames)
        self.assertEqual(info["video_fps"], 30)
        self.assertEqual(calls["count"], 2)


if __name__ == "__main__":
    unittest.main()
