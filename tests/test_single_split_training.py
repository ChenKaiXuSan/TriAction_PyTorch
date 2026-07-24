import unittest

from project.main import normalize_dataset_split


class SingleSplitTrainingTests(unittest.TestCase):
    def test_normalize_dataset_split_accepts_single_split(self):
        split = {"train": ["train-sample"], "val": ["val-sample"]}

        self.assertIs(normalize_dataset_split(split), split)

    def test_normalize_dataset_split_unwraps_legacy_single_fold(self):
        split = {"train": ["train-sample"], "val": ["val-sample"]}

        self.assertEqual(normalize_dataset_split({0: split}), split)

    def test_normalize_dataset_split_rejects_multiple_folds(self):
        fold_splits = {
            0: {"train": ["train-0"], "val": ["val-0"]},
            1: {"train": ["train-1"], "val": ["val-1"]},
        }

        with self.assertRaisesRegex(ValueError, "single dataset split"):
            normalize_dataset_split(fold_splits)


if __name__ == "__main__":
    unittest.main()
