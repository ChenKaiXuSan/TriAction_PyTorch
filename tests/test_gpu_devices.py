import unittest

from omegaconf import OmegaConf

from project.main import parse_train_devices


class TrainDeviceParsingTests(unittest.TestCase):
    def test_parses_single_gpu_integer(self):
        self.assertEqual(parse_train_devices(0), [0])

    def test_parses_gpu_list(self):
        self.assertEqual(parse_train_devices([0, 1]), [0, 1])

    def test_parses_omegaconf_list(self):
        cfg = OmegaConf.create({"gpu": [0, 1]})
        self.assertEqual(parse_train_devices(cfg.gpu), [0, 1])

    def test_parses_hydra_list_string(self):
        self.assertEqual(parse_train_devices("[0,1]"), [0, 1])

    def test_parses_comma_string(self):
        self.assertEqual(parse_train_devices("0,1"), [0, 1])


if __name__ == "__main__":
    unittest.main()
