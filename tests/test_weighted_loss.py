import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.trainer.losses import build_class_weights, weighted_cross_entropy


class WeightedLossTest(unittest.TestCase):
    def test_build_class_weights_uses_front_and_up_weights(self):
        cfg = OmegaConf.create(
            {
                "model": {"model_class_num": 5},
                "loss": {
                    "class_weights": {
                        "left": 1.0,
                        "right": 1.0,
                        "down": 1.0,
                        "up": 4.0,
                        "front": 0.2,
                    }
                },
            }
        )

        weights = build_class_weights(cfg, device=torch.device("cpu"))

        self.assertTrue(
            torch.equal(weights, torch.tensor([1.0, 1.0, 1.0, 4.0, 0.2]))
        )

    def test_weighted_cross_entropy_matches_torch_weighted_ce(self):
        logits = torch.tensor(
            [[2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0, 0.0]],
            dtype=torch.float32,
        )
        labels = torch.tensor([4, 3], dtype=torch.long)
        weights = torch.tensor([1.0, 1.0, 1.0, 4.0, 0.2], dtype=torch.float32)

        actual = weighted_cross_entropy(logits, labels, weights)
        expected = torch.nn.functional.cross_entropy(logits, labels, weight=weights)

        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
