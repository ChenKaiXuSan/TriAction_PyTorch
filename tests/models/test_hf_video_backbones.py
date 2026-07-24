import sys
import types

import torch
from omegaconf import OmegaConf


class _FakeConfig:
    def __init__(self, num_labels=2, hidden_size=8, **kwargs):
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class _FakeVideoModel(torch.nn.Module):
    from_pretrained_calls = []

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.last_pixel_values_shape = None

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.from_pretrained_calls.append((model_name, kwargs))
        return cls(
            _FakeConfig(
                num_labels=kwargs["num_labels"],
                hidden_size=kwargs.get("hidden_size", 8),
            )
        )

    def forward(self, pixel_values, output_hidden_states=False):
        self.last_pixel_values_shape = tuple(pixel_values.shape)
        features = pixel_values.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        features = features.repeat(1, self.config.hidden_size)
        logits = self.classifier(features)
        hidden_states = (features.unsqueeze(1),) if output_hidden_states else None
        return _FakeOutput(logits=logits, hidden_states=hidden_states)


def _install_fake_transformers(monkeypatch):
    module = types.ModuleType("transformers")
    module.VideoMAEConfig = _FakeConfig
    module.VideoMAEForVideoClassification = _FakeVideoModel
    module.VivitConfig = _FakeConfig
    module.VivitForVideoClassification = _FakeVideoModel
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


def _cfg(backbone, pretrained=False, extra_model=None):
    model_cfg = {
        "backbone": backbone,
        "model_class_num": 4,
        "hf_video_pretrained": pretrained,
        "hf_video_hidden_size": 8,
        "videomae_model_name": "fake/videomae",
        "vivit_model_name": "fake/vivit",
    }
    if extra_model:
        model_cfg.update(extra_model)
    return OmegaConf.create(
        {
            "model": model_cfg
        }
    )


def test_videomae_backbone_accepts_project_video_layout(monkeypatch):
    _install_fake_transformers(monkeypatch)
    from project.models.hf_video_backbone import VideoMAEBackbone

    model = VideoMAEBackbone(_cfg("videomae"))
    video = torch.randn(2, 3, 5, 16, 16)

    logits = model(video)
    logits.sum().backward()

    assert logits.shape == (2, 4)
    assert model.model.last_pixel_values_shape == (2, 5, 3, 16, 16)
    assert model.feature_dim == 8
    assert model.model.classifier.weight.grad is not None


def test_vivit_backbone_can_return_features(monkeypatch):
    _install_fake_transformers(monkeypatch)
    from project.models.hf_video_backbone import VivitBackbone

    model = VivitBackbone(_cfg("vivit"))
    video = torch.randn(2, 3, 5, 16, 16)

    features = model.forward_features(video)

    assert features.shape == (2, 8)


def test_hf_video_backbone_uses_pretrained_model_name(monkeypatch):
    fake_transformers = _install_fake_transformers(monkeypatch)
    from project.models.hf_video_backbone import VideoMAEBackbone

    model = VideoMAEBackbone(_cfg("videomae", pretrained=True))

    assert isinstance(model.model, _FakeVideoModel)
    assert fake_transformers.VideoMAEForVideoClassification.from_pretrained_calls[-1][0] == "fake/videomae"


def test_hf_video_backbone_passes_family_specific_revision(monkeypatch):
    fake_transformers = _install_fake_transformers(monkeypatch)
    from project.models.hf_video_backbone import VivitBackbone

    model = VivitBackbone(
        _cfg(
            "vivit",
            pretrained=True,
            extra_model={"vivit_model_revision": "refs/pr/3"},
        )
    )

    assert isinstance(model.model, _FakeVideoModel)
    assert fake_transformers.VivitForVideoClassification.from_pretrained_calls[-1][1][
        "revision"
    ] == "refs/pr/3"
