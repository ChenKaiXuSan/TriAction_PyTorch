import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.utils.visualize_annotation_distribution import build_summary


def _write_label(path, annotators):
    payload = {"annotations": []}
    for annotator, labels in annotators:
        payload["annotations"].append(
            {
                "annotator": annotator,
                "videoLabels": [
                    {
                        "timelinelabels": [label],
                        "ranges": [{"start": start, "end": end}],
                    }
                    for label, start, end in labels
                ],
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


class AnnotationDistributionTest(unittest.TestCase):
    def test_build_summary_counts_labels_and_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            label_dir = tmp_path / "label"
            label_dir.mkdir()
            _write_label(
                label_dir / "person_01_day_high_h265.json",
                [
                    (3, [("left", 10, 20), ("right_up", 30, 40)]),
                    (4, [("left_down", 11, 22), ("up", 50, 60)]),
                ],
            )
            _write_label(
                label_dir / "person_02_night_low_h265.json",
                [
                    (3, [("down", 5, 15)]),
                ],
            )

            split_file = tmp_path / "split_mid_end.json"
            split_file.write_text(
                json.dumps(
                    [
                        {
                            "video": "/data/local-files/?d=mydata/drive_data/person_01_day_high_h265.mp4",
                            "videoLabels": [
                                {
                                    "timelinelabels": ["start"],
                                    "ranges": [{"start": 1, "end": 1}],
                                },
                                {
                                    "timelinelabels": ["mid"],
                                    "ranges": [{"start": 50, "end": 50}],
                                },
                                {
                                    "timelinelabels": ["end"],
                                    "ranges": [{"start": 100, "end": 100}],
                                },
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            summary = build_summary(label_dir, split_file)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["annotation_count"], 3)
        self.assertEqual(
            summary["raw_label_counts"],
            {
                "down": 1,
                "left": 1,
                "left_down": 1,
                "right_up": 1,
                "up": 1,
            },
        )
        self.assertEqual(
            summary["coarse_label_counts"],
            {
                "down": 1,
                "left": 2,
                "right": 1,
                "up": 1,
            },
        )
        self.assertEqual(summary["annotator_counts"], {3: 3, 4: 2})
        self.assertEqual(summary["split_marker_count"], 1)
        self.assertEqual(summary["split_duration_stats"]["min"], 99)


if __name__ == "__main__":
    unittest.main()
