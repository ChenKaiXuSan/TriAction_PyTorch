import unittest

from omegaconf import OmegaConf

from project.dataloader.data_loader import DriverDataModule


class DataloaderWorkerOptionTests(unittest.TestCase):
    def test_disables_persistent_workers_when_num_workers_is_zero(self):
        cfg = OmegaConf.create(
            {
                "data": {
                    "num_workers": 0,
                    "val_num_workers": 0,
                    "img_size": 224,
                    "uniform_temporal_subsample_num": 8,
                    "batch_size": 1,
                    "max_video_frames": 1000,
                },
                "model": {"model_class_num": 5, "input_type": "rgb"},
                "train": {"view_name": ["front"]},
                "experiment": "test",
                "paths": {"start_mid_end_path": "dummy.json"},
            }
        )
        dm = DriverDataModule(cfg, {"train": [], "val": []})

        self.assertEqual(dm._worker_kwargs(), {})

    def test_sets_prefetch_factor_for_worker_processes(self):
        cfg = OmegaConf.create(
            {
                "data": {
                    "num_workers": 4,
                    "val_num_workers": 1,
                    "prefetch_factor": 1,
                    "img_size": 224,
                    "uniform_temporal_subsample_num": 8,
                    "batch_size": 1,
                    "max_video_frames": 1000,
                },
                "model": {"model_class_num": 5, "input_type": "rgb"},
                "train": {"view_name": ["front"]},
                "experiment": "test",
                "paths": {"start_mid_end_path": "dummy.json"},
            }
        )
        dm = DriverDataModule(cfg, {"train": [], "val": []})

        self.assertEqual(
            dm._worker_kwargs(4),
            {"persistent_workers": True, "prefetch_factor": 1},
        )

    def test_uses_separate_validation_worker_count(self):
        cfg = OmegaConf.create(
            {
                "data": {
                    "num_workers": 4,
                    "val_num_workers": 1,
                    "test_num_workers": 1,
                    "prefetch_factor": 1,
                    "img_size": 224,
                    "uniform_temporal_subsample_num": 8,
                    "batch_size": 1,
                    "max_video_frames": 1000,
                },
                "model": {"model_class_num": 5, "input_type": "rgb"},
                "train": {"view_name": ["front"]},
                "experiment": "test",
                "paths": {"start_mid_end_path": "dummy.json"},
            }
        )
        dm = DriverDataModule(cfg, {"train": [], "val": []})

        self.assertEqual(dm._num_workers, 4)
        self.assertEqual(dm._val_num_workers, 1)
        self.assertEqual(dm._test_num_workers, 1)


if __name__ == "__main__":
    unittest.main()
