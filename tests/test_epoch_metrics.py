"""Epoch metrics must be accumulated over the epoch, not averaged per batch.

With segment batching the batch size is 2, where per-batch macro F1 is wildly
biased; the 2026-07-26 runs reported 0.52 where the true value was 0.35.
"""

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassF1Score

from project.trainer.metrics import build_stage_metrics

NUM_CLASSES = 4


def _imbalanced_data(n=64):
    torch.manual_seed(0)
    # skewed labels, like the real split (left-heavy), and a mediocre predictor
    labels = torch.cat([torch.zeros(n // 2), torch.ones(n // 4), torch.full((n // 4,), 2.0)]).long()
    probs = torch.rand(n, NUM_CLASSES)
    probs[torch.arange(n), labels] += 0.6  # partially correct
    return probs / probs.sum(1, keepdim=True), labels


class _Probe(LightningModule):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(NUM_CLASSES, NUM_CLASSES)
        _, _, self.test_metrics = build_stage_metrics(NUM_CLASSES)

    def test_step(self, batch, batch_idx):
        probs, label = batch
        self.test_metrics(probs, label)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True, batch_size=label.size(0))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_logged_epoch_f1_matches_whole_epoch_computation():
    probs, labels = _imbalanced_data()
    loader = DataLoader(TensorDataset(probs, labels), batch_size=2)

    trainer = Trainer(accelerator="cpu", logger=False, enable_progress_bar=False,
                      enable_checkpointing=False, enable_model_summary=False)
    trainer.test(_Probe(), loader, verbose=False)
    logged = float(trainer.callback_metrics["test/video_f1_score_epoch"])

    expected = float(MulticlassF1Score(num_classes=NUM_CLASSES)(probs, labels))
    per_batch_avg = float(torch.tensor([
        MulticlassF1Score(num_classes=NUM_CLASSES)(probs[i:i + 2], labels[i:i + 2])
        for i in range(0, len(labels), 2)
    ]).mean())

    assert abs(logged - expected) < 1e-4, f"logged {logged} != epoch-level {expected}"
    # guard that the test would actually catch the old bug
    assert abs(per_batch_avg - expected) > 0.05, "per-batch average is not distinguishable here"


def test_stage_metrics_are_independent_instances():
    train, val, test = build_stage_metrics(NUM_CLASSES)
    probs, labels = _imbalanced_data(8)
    train(probs, labels)
    assert val["video_acc"].update_count == 0, "val metrics must not accumulate train batches"
    assert test["video_acc"].update_count == 0
