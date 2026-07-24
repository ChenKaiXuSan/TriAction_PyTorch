import unittest
from pathlib import Path

from project.cross_validation import DefineCrossValidation
from project.map_config import VideoSample


def sample(person_id: str) -> VideoSample:
    return VideoSample(
        person_id=person_id,
        env_folder="day_high",
        env_key="day_high",
        label_path=Path(f"/labels/person_{person_id}_day_high_h265.json"),
        videos={"front": Path(f"/videos/{person_id}/front.mp4")},
    )


class SingleSplitCrossValidationTests(unittest.TestCase):
    def test_build_single_split_uses_all_samples_as_train_initially(self):
        samples = [sample("01"), sample("02"), sample("03")]

        split = DefineCrossValidation.build_single_split(samples)

        self.assertEqual(split, {"train": samples, "val": []})


if __name__ == "__main__":
    unittest.main()
