import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rwf2000.evaluate_baseline import evaluate_test_fixed_threshold, finalize_validation
from rwf2000.models import ResNet18TemporalAverage
from rwf2000.train_baseline import build_optimizer, save_checkpoint
from rwf2000.train_config import TrainConfig


class TinyBackbone(nn.Module):
    feature_dim = 2

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def forward(self, frames):
        return self.projection(frames.mean(dim=(2, 3)))


class Loader(list):
    def __init__(self, batches, derived_split):
        super().__init__(batches)
        self.dataset = type("Dataset", (), {})()
        source_split = "val" if derived_split == "test" else "train"
        self.dataset.records = [
            type(
                "Record",
                (),
                {"derived_split": derived_split, "source_split": source_split},
            )()
        ]


def _model():
    return ResNet18TemporalAverage(
        weights=None,
        backbone=TinyBackbone(),
        feature_dim=2,
        freeze_backbone=True,
    )


def _batch(labels):
    values = torch.tensor(labels, dtype=torch.float32)
    return {
        "frames": values.reshape(-1, 1, 1, 1, 1).repeat(1, 1, 3, 1, 1),
        "label": values,
        "metadata": {
            "clip_id": [f"clip-{index}" for index in range(len(labels))],
            "relative_path": [f"video-{index}.mp4" for index in range(len(labels))],
        },
    }


def _checkpoint(tmp_path, model):
    optimizer = build_optimizer(model, TrainConfig(weights=None))
    path = tmp_path / "best_checkpoint.pth"
    run_config = TrainConfig(weights=None).to_dict()
    run_config["manifest_sha256"] = "hash"
    run_config.update({"architecture": "ResNet18TemporalAverage", "feature_dim": 2})
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_val_loss=0.5,
        run_config=run_config,
        manifest_sha256="hash",
        environment={"torch_version": str(torch.__version__)},
    )
    return path


def test_finalize_validation_writes_predictions_metrics_and_threshold(tmp_path):
    model = _model()
    checkpoint = _checkpoint(tmp_path, model)
    loader = Loader([_batch([0.0, 1.0])], "val")

    result = finalize_validation(model, checkpoint, loader, tmp_path, manifest_sha256="hash")

    assert Path(result["predictions"]).is_file()
    assert Path(result["metrics"]).is_file()
    assert Path(result["threshold"]).is_file()
    threshold = json.loads(Path(result["threshold"]).read_text(encoding="utf-8"))
    assert threshold["selection_split"] == "validation"
    assert threshold["schema"] == "rwf2000-selected-threshold-v1"
    assert "rule" in threshold
    assert threshold["threshold_rule"] == threshold["rule"]
    assert threshold["manifest_sha256"] == "hash"
    assert threshold["checkpoint_sha256"] == threshold["run_id"]
    assert threshold["best_epoch"] == 1
    assert threshold["architecture"] == "ResNet18TemporalAverage"
    assert threshold["architecture_contract"]["feature_dim"] == 2
    assert threshold["validation_predictions_sha256"]
    prediction_text = Path(result["predictions"]).read_text(encoding="utf-8")
    assert "clip_id,relative_path,label,logit,probability,threshold,predicted_label" in prediction_text
    assert "clip-0" in prediction_text
    with Path(result["predictions"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(
        int(row["predicted_label"]) == int(float(row["probability"]) >= float(row["threshold"]))
        for row in rows
    )


def test_fixed_test_evaluation_never_selects_threshold(tmp_path, monkeypatch):
    model = _model()
    checkpoint = _checkpoint(tmp_path, model)
    validation_loader = Loader([_batch([0.0, 1.0])], "val")
    validation = finalize_validation(
        model, checkpoint, validation_loader, tmp_path, manifest_sha256="hash"
    )
    threshold_path = validation["threshold"]
    loader = Loader([_batch([0.0, 1.0])], "test")

    def forbidden(*args, **kwargs):
        raise AssertionError("test evaluation must not select a threshold")

    monkeypatch.setattr("rwf2000.evaluate_baseline.select_threshold", forbidden)
    result = evaluate_test_fixed_threshold(
        model,
        checkpoint,
        threshold_path,
        loader,
        tmp_path,
        manifest_sha256="hash",
    )

    assert Path(result["predictions"]).is_file()
    assert Path(result["metrics"]).is_file()
    assert Path(result["confusion_matrix"]).is_file()
    assert result["threshold_value"] == json.loads(
        Path(threshold_path).read_text(encoding="utf-8")
    )["threshold"]
    metrics = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))
    assert metrics["manifest_sha256"] == "hash"
    assert metrics["threshold_sha256"]
    assert metrics["evaluation_loss"] >= 0
    assert metrics["best_val_loss"] == 0.5


def test_fixed_test_evaluation_rejects_non_test_records(tmp_path):
    model = _model()
    checkpoint = _checkpoint(tmp_path, model)
    validation = finalize_validation(
        model,
        checkpoint,
        Loader([_batch([0.0, 1.0])], "val"),
        tmp_path,
        manifest_sha256="hash",
    )
    threshold_path = validation["threshold"]
    loader = Loader([_batch([0.0, 1.0])], "val")

    with pytest.raises(ValueError, match="derived split"):
        evaluate_test_fixed_threshold(
            model,
            checkpoint,
            threshold_path,
            loader,
            tmp_path,
            manifest_sha256="hash",
        )


def test_fixed_test_evaluation_rejects_provenance_mismatch_and_missing_fields(tmp_path):
    model = _model()
    checkpoint = _checkpoint(tmp_path, model)
    validation = finalize_validation(
        model,
        checkpoint,
        Loader([_batch([0.0, 1.0])], "val"),
        tmp_path,
        manifest_sha256="hash",
    )
    threshold_payload = json.loads(Path(validation["threshold"]).read_text(encoding="utf-8"))
    threshold_payload["manifest_sha256"] = "different"
    mismatched = tmp_path / "mismatched_threshold.json"
    mismatched.write_text(json.dumps(threshold_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="threshold artifact manifest_sha256"):
        evaluate_test_fixed_threshold(
            model,
            checkpoint,
            mismatched,
            Loader([_batch([0.0, 1.0])], "test"),
            tmp_path / "mismatch-output",
            manifest_sha256="hash",
        )

    missing = tmp_path / "missing_threshold.json"
    missing.write_text(json.dumps({"threshold": 0.5, "selection_split": "validation"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        evaluate_test_fixed_threshold(
            model,
            checkpoint,
            missing,
            Loader([_batch([0.0, 1.0])], "test"),
            tmp_path / "missing-output",
            manifest_sha256="hash",
        )


def test_final_test_artifacts_are_not_overwritten_by_default(tmp_path):
    model = _model()
    checkpoint = _checkpoint(tmp_path, model)
    validation = finalize_validation(
        model,
        checkpoint,
        Loader([_batch([0.0, 1.0])], "val"),
        tmp_path,
        manifest_sha256="hash",
    )
    loader = Loader([_batch([0.0, 1.0])], "test")
    output = tmp_path / "test-output"
    evaluate_test_fixed_threshold(
        model,
        checkpoint,
        validation["threshold"],
        loader,
        output,
        manifest_sha256="hash",
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate_test_fixed_threshold(
            model,
            checkpoint,
            validation["threshold"],
            loader,
            output,
            manifest_sha256="hash",
        )


def test_fixed_test_evaluation_requires_manifest_records(tmp_path):
    model = _model()
    checkpoint = _checkpoint(tmp_path, model)
    validation = finalize_validation(
        model,
        checkpoint,
        Loader([_batch([0.0, 1.0])], "val"),
        tmp_path,
        manifest_sha256="hash",
    )

    class MissingRecordsLoader(list):
        def __init__(self):
            super().__init__([_batch([0.0, 1.0])])
            self.dataset = SimpleNamespace()

    with pytest.raises(ValueError, match="expose manifest records"):
        evaluate_test_fixed_threshold(
            model,
            checkpoint,
            validation["threshold"],
            MissingRecordsLoader(),
            tmp_path / "missing-records",
            manifest_sha256="hash",
        )


def test_finalize_validation_requires_matching_runtime_manifest(tmp_path):
    model = _model()
    checkpoint = _checkpoint(tmp_path, model)
    loader = Loader([_batch([0.0, 1.0])], "val")

    with pytest.raises(ValueError, match="runtime manifest_sha256 is required"):
        finalize_validation(model, checkpoint, loader, tmp_path / "missing")
    with pytest.raises(ValueError, match="does not match checkpoint provenance"):
        finalize_validation(
            model,
            checkpoint,
            loader,
            tmp_path / "mismatch",
            manifest_sha256="different",
        )


def test_threshold_artifact_rejects_tampered_selected_evidence(tmp_path):
    model = _model()
    checkpoint = _checkpoint(tmp_path, model)
    validation = finalize_validation(
        model,
        checkpoint,
        Loader([_batch([0.0, 1.0])], "val"),
        tmp_path,
        manifest_sha256="hash",
    )
    payload = json.loads(Path(validation["threshold"]).read_text(encoding="utf-8"))
    payload["selected_f1"] = 1.0 - float(payload["selected_f1"])
    tampered = tmp_path / "tampered_threshold.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="selected_f1 disagrees"):
        evaluate_test_fixed_threshold(
            model,
            checkpoint,
            tampered,
            Loader([_batch([0.0, 1.0])], "test"),
            tmp_path / "tampered-output",
            manifest_sha256="hash",
        )


def test_threshold_artifact_links_architecture_and_validation_predictions(tmp_path):
    model = _model()
    checkpoint = _checkpoint(tmp_path, model)
    validation = finalize_validation(
        model,
        checkpoint,
        Loader([_batch([0.0, 1.0])], "val"),
        tmp_path,
        manifest_sha256="hash",
    )
    payload = json.loads(Path(validation["threshold"]).read_text(encoding="utf-8"))
    payload["architecture"] = "ResNet18TemporalTransformer"
    tampered_architecture = tmp_path / "tampered_architecture.json"
    tampered_architecture.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture does not match"):
        evaluate_test_fixed_threshold(
            model,
            checkpoint,
            tampered_architecture,
            Loader([_batch([0.0, 1.0])], "test"),
            tmp_path / "tampered-architecture-output",
            manifest_sha256="hash",
        )

    original_threshold = validation["threshold"]
    predictions = tmp_path / "validation_predictions.csv"
    predictions.write_text(
        predictions.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation_predictions_sha256 does not match"):
        evaluate_test_fixed_threshold(
            model,
            checkpoint,
            original_threshold,
            Loader([_batch([0.0, 1.0])], "test"),
            tmp_path / "tampered-predictions-output",
            manifest_sha256="hash",
        )
