import inspect
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rwf2000.manifest import ManifestRecord
from rwf2000.models import ResNet18TemporalAverage
from rwf2000.train_baseline import (
    build_optimizer,
    evaluate_loader,
    fit,
    load_checkpoint,
    save_checkpoint,
    set_global_seed,
    train_one_epoch,
)
from rwf2000.train_config import TrainConfig


class TinyBackbone(nn.Module):
    feature_dim = 2

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def forward(self, frames):
        return self.projection(frames.mean(dim=(2, 3)))


class EpochDataset:
    def __init__(self, records=None):
        self.records = records
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class Loader(list):
    def __init__(self, batches, records=None):
        super().__init__(batches)
        self.dataset = EpochDataset(records)


def _batch(values, labels):
    values = torch.as_tensor(values, dtype=torch.float32)
    return {
        "frames": values.reshape(-1, 1, 1, 1, 1).repeat(1, 1, 3, 1, 1),
        "label": torch.as_tensor(labels, dtype=torch.float32),
    }


def _model(freeze_backbone=True):
    return ResNet18TemporalAverage(
        weights=None,
        backbone=TinyBackbone(),
        feature_dim=2,
        freeze_backbone=freeze_backbone,
    )


def _config(**overrides):
    values = {
        "epochs": 1,
        "batch_size": 2,
        "feature_dim": 2,
        "weights": None,
        "purpose": "local_smoke_only",
    }
    values.update(overrides)
    return TrainConfig(**values)


def _record(derived_split):
    return ManifestRecord(
        schema_version="rwf2000-manifest-v1",
        seed=42,
        val_fraction=0.2,
        label_map="{}",
        relative_path=f"{derived_split}/clip.mp4",
        clip_id="a" * 64,
        source_split="train",
        derived_split=derived_split,
        label=0,
        label_name="Non-violence",
    )


class ZeroLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, frames):
        return self.anchor.expand(frames.shape[0]) * 0


def test_train_config_is_frozen_and_json_round_trips(tmp_path):
    config = _config(learning_rate=0.002, height=32, width=32)
    path = tmp_path / "config.json"
    config.save_json(path)

    assert TrainConfig.load_json(path) == config
    with pytest.raises(Exception):
        config.epochs = 2

    with pytest.raises(ValueError, match="None or.*DEFAULT"):
        TrainConfig(weights="IMAGENET1K_V1")


@pytest.mark.parametrize(
    "config_name, purpose",
    [
        ("rwf2000_resnet18_avg_local_smoke.json", "local_smoke_only"),
        ("rwf2000_resnet18_avg_kaggle_pilot.json", "kaggle_pilot_only_not_final"),
        ("rwf2000_resnet18_avg_kaggle_full.json", "kaggle_full_baseline"),
        ("rwf2000_resnet18_transformer_local_smoke.json", "local_smoke_only"),
        (
            "rwf2000_resnet18_transformer_kaggle_pilot.json",
            "phase3_kaggle_pilot_only_not_final",
        ),
        (
            "rwf2000_resnet18_transformer_kaggle_full.json",
            "phase3_kaggle_full_transformer",
        ),
    ],
)
def test_committed_training_configs_are_valid(config_name, purpose):
    config_path = Path(__file__).parents[1] / "configs" / config_name

    config = TrainConfig.load_json(config_path)

    assert config.purpose == purpose
    if "transformer" in config_name:
        assert config.architecture == "ResNet18TemporalTransformer"
        assert config.architecture_version == "v1"
        assert config.feature_dim == 512
        assert config.num_frames == 16
        assert config.num_layers == 2
        assert config.nhead == 4
        assert config.dim_feedforward == 1024
        assert config.dropout == 0.1
        assert config.positional_encoding_type == "sinusoidal"
        assert config.max_sequence_length == 16
        assert config.norm_first is True
        assert config.activation == "gelu"
        assert config.temporal_pooling == "mean"
        assert config.expected_manifest_sha256 is not None


def test_set_global_seed_reproduces_torch_and_numpy_streams():
    policy = set_global_seed(123, deterministic=True)
    first_torch = torch.rand(3)
    first_numpy = __import__("numpy").random.rand(3)
    set_global_seed(123, deterministic=True)

    assert torch.equal(first_torch, torch.rand(3))
    assert (first_numpy == __import__("numpy").random.rand(3)).all()
    assert policy["deterministic"] is True


def test_build_optimizer_uses_trainable_parameters_only():
    model = _model(freeze_backbone=True)
    optimizer = build_optimizer(model, _config())

    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.classifier.parameters()}

    with pytest.raises(ValueError, match="no trainable"):
        build_optimizer(_model(freeze_backbone=True).backbone, _config())


def test_train_loss_is_sample_weighted_and_updates_classifier():
    model = _model(freeze_backbone=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [_batch([0.0, 1.0], [0.0, 1.0]), _batch([2.0], [1.0])]
    before = model.classifier.weight.detach().clone()

    result = train_one_epoch(model, loader, optimizer)

    assert result["num_samples"] == 3
    assert result["loss"] > 0
    assert not torch.equal(before, model.classifier.weight)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_evaluate_loader_is_no_grad_and_collects_predictions():
    model = _model(freeze_backbone=False)
    loader = [
        {
            **_batch([0.0, 1.0], [0.0, 1.0]),
            "metadata": {"clip_id": ["clip-a", "clip-b"], "relative_path": ["a.mp4", "b.mp4"]},
        }
    ]

    result = evaluate_loader(model, loader)

    assert not model.training
    assert result["labels"].shape == (2,)
    assert result["logits"].shape == (2,)
    assert result["probabilities"].shape == (2,)
    assert result["records"] == [
        {"clip_id": "clip-a", "relative_path": "a.mp4"},
        {"clip_id": "clip-b", "relative_path": "b.mp4"},
    ]
    assert not result["logits"].requires_grad


def test_fit_rejects_mixed_or_wrong_real_derived_splits(tmp_path):
    model = _model()
    train_loader = Loader([_batch([0.0], [0.0])], records=[_record("val")])
    val_loader = Loader([_batch([0.0], [0.0])], records=[_record("val")])

    with pytest.raises(ValueError, match="train.*derived split"):
        fit(model, train_loader, val_loader, _config(), tmp_path)


def test_fit_rejects_source_split_mismatch_and_missing_records(tmp_path):
    wrong_source = replace(_record("train"), source_split="val")
    train_loader = Loader([_batch([0.0], [0.0])], records=[wrong_source])
    val_loader = Loader([_batch([0.0], [0.0])], records=[_record("val")])
    with pytest.raises(ValueError, match="source split"):
        fit(_model(), train_loader, val_loader, _config(), tmp_path / "source")

    class MissingRecordsLoader(list):
        def __init__(self):
            super().__init__([_batch([0.0], [0.0])])
            self.dataset = SimpleNamespace()

    with pytest.raises(ValueError, match="expose manifest records"):
        fit(
            _model(),
            MissingRecordsLoader(),
            val_loader,
            _config(),
            tmp_path / "missing",
        )


def test_fit_signature_has_no_test_loader():
    assert "test_loader" not in inspect.signature(fit).parameters


def test_fit_calls_set_epoch_and_earliest_tie_wins(tmp_path):
    model = ZeroLogitModel()
    train_loader = Loader([_batch([0.0], [0.0])], records=[_record("train")])
    val_loader = Loader([_batch([0.0], [0.0])], records=[_record("val")])
    result = fit(
        model,
        train_loader,
        val_loader,
        _config(epochs=2, learning_rate=1e-3),
        tmp_path,
    )

    assert train_loader.dataset.epochs == [0, 1]
    assert result["best_epoch"] == 1
    assert result["history"][0]["val_loss"] == result["history"][1]["val_loss"]
    try:
        checkpoint = torch.load(
            tmp_path / "best_checkpoint.pth",
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        pytest.skip("installed torch does not support weights_only checkpoint loading")
    assert checkpoint["epoch"] == 1
    assert isinstance(checkpoint["environment"]["torch_version"], str)
    assert (tmp_path / "run_config.json").is_file()
    assert (tmp_path / "environment.json").is_file()
    assert (tmp_path / "history.csv").is_file()


def test_checkpoint_round_trip_restores_identical_logits(tmp_path):
    model = _model()
    optimizer = build_optimizer(model, _config())
    frames = torch.randn(2, 1, 3, 1, 1)
    before = model(frames).detach()
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_val_loss=0.25,
        run_config={
            **_config().to_dict(),
            "architecture": "ResNet18TemporalAverage",
            "feature_dim": 2,
        },
        manifest_sha256="manifest-hash",
        environment={"torch_version": torch.__version__},
    )

    restored = _model()
    restored_optimizer = build_optimizer(restored, _config())
    payload = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        map_location="cpu",
        restore_rng=False,
    )

    assert torch.equal(before, restored(frames).detach())
    assert payload["epoch"] == 3
    assert payload["manifest_sha256"] == "manifest-hash"
    assert payload["selection_metric"] == "val_loss"
