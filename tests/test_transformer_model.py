from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from rwf2000.evaluate_baseline import _checkpoint_provenance
from rwf2000.models import (
    ResNet18TemporalAverage,
    ResNet18TemporalTransformer,
    build_model_from_config,
)
from rwf2000.train_baseline import build_optimizer, fit, load_checkpoint, save_checkpoint
from rwf2000.train_config import EXPECTED_RWF2000_MANIFEST_SHA256, TrainConfig


class TinyFeatureBackbone(nn.Module):
    feature_dim = 8

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, self.feature_dim)

    def forward(self, frames):
        return self.projection(frames.mean(dim=(2, 3)))


class BatchNormFeatureBackbone(nn.Module):
    feature_dim = 8

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, frames):
        return self.features(frames)


def _model(*, backbone=None, freeze_backbone=True):
    return ResNet18TemporalTransformer(
        weights=None,
        backbone=backbone or TinyFeatureBackbone(),
        feature_dim=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        max_sequence_length=4,
        freeze_backbone=freeze_backbone,
    )


def test_transformer_shapes_finite_and_sequence_limit():
    model = _model()
    for batch_size in (1, 3):
        logits = model(torch.randn(batch_size, 4, 3, 8, 8))
        assert logits.shape == (batch_size,)
        assert torch.isfinite(logits).all()

    with pytest.raises(ValueError, match="max_sequence_length"):
        model(torch.randn(1, 5, 3, 8, 8))
    with pytest.raises(ValueError, match="3 RGB"):
        model(torch.randn(1, 4, 1, 8, 8))


def test_sinusoidal_encoding_makes_temporal_order_observable():
    model = _model()
    model.eval()
    frames = torch.randn(1, 4, 3, 8, 8)
    forward = model(frames)
    reversed_order = model(frames.flip(1))
    assert not torch.allclose(forward, reversed_order)


def test_frozen_backbone_and_batchnorm_stay_eval_and_optimizer_updates_head():
    model = _model(backbone=BatchNormFeatureBackbone())
    model.train()
    before_running_mean = model.backbone.features[1].running_mean.detach().clone()
    before_transformer = next(model.transformer.parameters()).detach().clone()
    before_classifier = model.classifier.weight.detach().clone()
    optimizer = build_optimizer(
        model,
        TrainConfig(
            architecture="ResNet18TemporalTransformer",
            feature_dim=8,
            num_frames=4,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            max_sequence_length=4,
            weights=None,
        ),
    )
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert id(next(model.transformer.parameters())) in optimized
    assert id(next(model.classifier.parameters())) in optimized

    loss = nn.BCEWithLogitsLoss()(model(torch.randn(2, 4, 3, 8, 8)), torch.tensor([0.0, 1.0]))
    loss.backward()
    optimizer.step()
    assert torch.equal(before_running_mean, model.backbone.features[1].running_mean)
    assert not torch.equal(before_transformer, next(model.transformer.parameters()))
    assert not torch.equal(before_classifier, model.classifier.weight)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_transformer_checkpoint_round_trip_preserves_logits(tmp_path):
    model = _model()
    model.eval()
    optimizer = build_optimizer(model, TrainConfig(weights=None))
    frames = torch.randn(2, 4, 3, 8, 8)
    before = model(frames).detach()
    path = tmp_path / "transformer.pth"
    run_config = TrainConfig(
        architecture="ResNet18TemporalTransformer",
        feature_dim=8,
        num_frames=4,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        max_sequence_length=4,
        weights=None,
    ).to_dict()
    run_config["manifest_sha256"] = "a" * 64
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_val_loss=0.5,
        run_config=run_config,
        manifest_sha256="a" * 64,
        environment={"torch_version": str(torch.__version__)},
    )
    restored = _model()
    load_checkpoint(path, model=restored, map_location="cpu", restore_rng=False)
    restored.eval()
    assert torch.equal(before, restored(frames).detach())


def test_phase3_config_round_trip_dispatch_and_invalid_values(tmp_path):
    source = TrainConfig.load_json(
        Path(__file__).parents[1] / "configs" / "rwf2000_resnet18_transformer_local_smoke.json"
    )
    path = tmp_path / "config.json"
    source.save_json(path)
    assert TrainConfig.load_json(path) == source
    model = build_model_from_config(source)
    assert isinstance(model, ResNet18TemporalTransformer)
    assert model.max_sequence_length == source.max_sequence_length
    legacy_baseline = build_model_from_config(
        {
            "architecture": "ResNet18TemporalAverage",
            "feature_dim": 8,
            "freeze_backbone": True,
            "weights": None,
        },
        backbone=TinyFeatureBackbone(),
    )
    assert isinstance(legacy_baseline, ResNet18TemporalAverage)

    with pytest.raises(ValueError, match="sinusoidal"):
        replace(source, positional_encoding_type="learned")
    with pytest.raises(ValueError, match="architecture"):
        replace(source, architecture="UnknownModel")
    with pytest.raises(ValueError, match="architecture_version"):
        replace(source, architecture_version="v2")
    with pytest.raises(ValueError, match="architecture_version"):
        build_model_from_config({**source.to_dict(), "architecture_version": "v2"})
    with pytest.raises(ValueError, match="max_sequence_length"):
        replace(source, max_sequence_length=source.num_frames - 1)


def test_fit_rejects_model_config_architecture_and_manifest_mismatches(tmp_path):
    model = _model()
    with pytest.raises(ValueError, match="architecture contract mismatch"):
        fit(
            model,
            None,
            None,
            TrainConfig(weights=None),
            tmp_path,
        )

    config = TrainConfig(
        architecture="ResNet18TemporalTransformer",
        feature_dim=8,
        num_frames=4,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        max_sequence_length=4,
        expected_manifest_sha256=EXPECTED_RWF2000_MANIFEST_SHA256,
        weights=None,
    )
    with pytest.raises(ValueError, match="expected_manifest_sha256"):
        fit(
            model,
            None,
            None,
            config,
            tmp_path,
            manifest_sha256="a" * 64,
        )


def test_transformer_checkpoint_cannot_load_as_baseline(tmp_path):
    model = _model()
    optimizer = build_optimizer(
        model,
        TrainConfig(
            architecture="ResNet18TemporalTransformer",
            feature_dim=8,
            num_frames=4,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            max_sequence_length=4,
            weights=None,
        ),
    )
    path = tmp_path / "transformer.pth"
    run_config = TrainConfig(
        architecture="ResNet18TemporalTransformer",
        feature_dim=8,
        num_frames=4,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        max_sequence_length=4,
        weights=None,
    ).to_dict()
    run_config["manifest_sha256"] = "a" * 64
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_val_loss=0.5,
        run_config=run_config,
        manifest_sha256="a" * 64,
        environment={"torch_version": str(torch.__version__)},
    )
    wrong_attention_contract = ResNet18TemporalTransformer(
        weights=None,
        backbone=TinyFeatureBackbone(),
        feature_dim=8,
        nhead=4,
        dim_feedforward=16,
        dropout=0.0,
        max_sequence_length=4,
    )
    with pytest.raises(ValueError, match="nhead"):
        load_checkpoint(
            path,
            model=wrong_attention_contract,
            map_location="cpu",
            restore_rng=False,
        )
    baseline = ResNet18TemporalAverage(
        weights=None,
        backbone=TinyFeatureBackbone(),
        feature_dim=8,
    )
    with pytest.raises(ValueError, match="architecture contract mismatch"):
        load_checkpoint(path, model=baseline, map_location="cpu", restore_rng=False)


def test_transformer_provenance_rejects_unknown_architecture_and_manifest_mismatch(tmp_path):
    payload = {
        "manifest_sha256": "a" * 64,
        "run_config": {
            "architecture": "UnknownModel",
            "feature_dim": 8,
            "manifest_sha256": "a" * 64,
        },
        "selection_metric": "val_loss",
        "selection_mode": "min",
        "checkpoint_rule": "min_validation_loss_earliest_tie",
        "best_val_loss": 0.5,
        "epoch": 1,
    }
    path = tmp_path / "payload.pth"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="architecture"):
        _checkpoint_provenance(path, payload)

    transformer = TrainConfig(
        architecture="ResNet18TemporalTransformer",
        feature_dim=8,
        num_frames=4,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        max_sequence_length=4,
        expected_manifest_sha256=EXPECTED_RWF2000_MANIFEST_SHA256,
        weights=None,
    ).to_dict()
    payload["run_config"] = {**transformer, "manifest_sha256": "a" * 64}
    with pytest.raises(ValueError, match="expected_manifest_sha256"):
        _checkpoint_provenance(path, payload)
