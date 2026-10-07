"""Validation finalization and fixed-threshold test evaluation."""

from __future__ import annotations

import csv
import json
import math
import numbers
from pathlib import Path
from typing import Any, Mapping

import torch

from .metrics import compute_binary_metrics, select_threshold
from .models import architecture_contract_from_config
from .train_baseline import _validate_loader_split, evaluate_loader, load_checkpoint, sha256_file


SUPPORTED_ARCHITECTURES = {
    "ResNet18TemporalAverage",
    "ResNet18TemporalTransformer",
}


PREDICTION_FIELDS = (
    "clip_id",
    "relative_path",
    "label",
    "logit",
    "probability",
    "threshold",
    "predicted_label",
)

# This is intentionally a small, explicit contract.  The validation artifact
# is evidence for the selected threshold, not a free-form configuration file.
THRESHOLD_ARTIFACT_SCHEMA = "rwf2000-selected-threshold-v1"
THRESHOLD_RULE = (
    "maximize Violence F1; tie-break by Violence recall; remaining ties choose smallest threshold"
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_predictions(path: Path, result: Mapping[str, Any], threshold: float) -> Path:
    labels = result["labels"].tolist()
    logits = result["logits"].tolist()
    probabilities = result["probabilities"].tolist()
    records = list(result.get("records", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index, (label, logit, probability) in enumerate(zip(labels, logits, probabilities)):
            metadata = records[index] if index < len(records) else {}
            writer.writerow(
                {
                    "clip_id": metadata.get("clip_id", ""),
                    "relative_path": metadata.get("relative_path", ""),
                    "label": int(label),
                    "logit": float(logit),
                    "probability": float(probability),
                    "threshold": float(threshold),
                    "predicted_label": int(float(probability) >= threshold),
                }
            )
    return path


def _write_confusion_matrix(path: Path, matrix: list[list[int]], title: str) -> Path:
    # Plotting is lazy so Phase 1 commands do not import matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Non-violence", "Violence"],
        yticklabels=["Non-violence", "Violence"],
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row][column], ha="center", va="center")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _read_threshold_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("threshold artifact must contain a JSON object")
    if payload.get("schema") != THRESHOLD_ARTIFACT_SCHEMA:
        raise ValueError(
            f"threshold artifact schema must be {THRESHOLD_ARTIFACT_SCHEMA!r}"
        )
    if payload.get("selection_split") != "validation":
        raise ValueError("threshold artifact must state selection_split='validation'")
    if payload.get("rule") != THRESHOLD_RULE or payload.get("threshold_rule") != THRESHOLD_RULE:
        raise ValueError("threshold artifact has an unsupported threshold rule")
    if "threshold" not in payload:
        raise ValueError("threshold artifact is missing 'threshold'")
    threshold = payload["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise ValueError("threshold artifact must contain a numeric threshold")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold artifact value must be in [0, 1]")
    candidate_count = payload.get("candidate_count")
    candidates = payload.get("candidates")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
        or not isinstance(candidates, list)
        or len(candidates) != candidate_count
    ):
        raise ValueError("threshold artifact is missing valid candidate evidence")
    evidence: list[dict[str, float]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"threshold candidate {index} must be an object")
        if set(candidate) != {"threshold", "f1", "recall"}:
            raise ValueError(f"threshold candidate {index} has an invalid evidence schema")
        values: dict[str, float] = {}
        for field in ("threshold", "f1", "recall"):
            value = candidate[field]
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"threshold candidate {index} field {field!r} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"threshold candidate {index} field {field!r} must be finite")
            values[field] = value
        if not 0.0 <= values["threshold"] <= 1.0:
            raise ValueError("threshold candidate threshold must be in [0, 1]")
        if not 0.0 <= values["f1"] <= 1.0 or not 0.0 <= values["recall"] <= 1.0:
            raise ValueError("threshold candidate scores must be in [0, 1]")
        evidence.append(values)
    for field in ("selected_f1", "selected_recall"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"threshold artifact is missing numeric {field}")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold artifact {field} must be finite and in [0, 1]")
        payload[field] = value
    support = payload.get("support")
    if isinstance(support, bool) or not isinstance(support, int) or support <= 0:
        raise ValueError("threshold artifact is missing valid support evidence")
    expected = min(evidence, key=lambda item: (-item["f1"], -item["recall"], item["threshold"]))
    if not math.isclose(threshold, expected["threshold"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("threshold artifact selected threshold disagrees with candidate evidence")
    if not math.isclose(payload["selected_f1"], expected["f1"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("threshold artifact selected_f1 disagrees with candidate evidence")
    if not math.isclose(payload["selected_recall"], expected["recall"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("threshold artifact selected_recall disagrees with candidate evidence")
    payload["threshold"] = threshold
    return payload


def _read_threshold(path: str | Path) -> float:
    """Read only the numeric threshold for compatibility with callers."""

    return float(_read_threshold_artifact(path)["threshold"])


def _class_counts(labels: Any) -> dict[str, int]:
    values = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    return {
        "Non-violence": sum(int(float(value) == 0.0) for value in values),
        "Violence": sum(int(float(value) == 1.0) for value in values),
    }


def _checkpoint_provenance(checkpoint_path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_sha256 = sha256_file(checkpoint_path)
    manifest_sha256 = payload.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or not manifest_sha256:
        raise ValueError("checkpoint is missing a non-empty manifest_sha256")
    run_config = payload.get("run_config")
    if not isinstance(run_config, Mapping):
        raise ValueError("checkpoint is missing run_config provenance")
    if payload.get("selection_metric") != "val_loss":
        raise ValueError("checkpoint selection_metric must be 'val_loss'")
    if payload.get("selection_mode") != "min":
        raise ValueError("checkpoint selection_mode must be 'min'")
    if payload.get("checkpoint_rule") != "min_validation_loss_earliest_tie":
        raise ValueError("checkpoint checkpoint_rule is unsupported")
    best_val_loss = payload.get("best_val_loss")
    if (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, numbers.Real)
        or not math.isfinite(float(best_val_loss))
    ):
        raise ValueError("checkpoint is missing a finite best_val_loss")
    architecture = run_config.get("architecture")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "checkpoint architecture must be one of "
            f"{sorted(SUPPORTED_ARCHITECTURES)!r}"
        )
    feature_dim = run_config.get("feature_dim")
    if (
        isinstance(feature_dim, bool)
        or not isinstance(feature_dim, numbers.Integral)
        or int(feature_dim) <= 0
    ):
        raise ValueError("checkpoint feature_dim must be a positive integer")
    run_manifest = run_config.get("manifest_sha256")
    if not isinstance(run_manifest, str) or not run_manifest:
        raise ValueError("checkpoint run_config is missing a non-empty manifest_sha256")
    if run_manifest != manifest_sha256:
        raise ValueError("checkpoint manifest_sha256 disagrees with run_config")
    expected_manifest = run_config.get("expected_manifest_sha256")
    if expected_manifest is not None and expected_manifest != manifest_sha256:
        raise ValueError("checkpoint expected_manifest_sha256 disagrees with manifest_sha256")
    if architecture == "ResNet18TemporalTransformer":
        required = (
            "architecture_version",
            "num_layers",
            "nhead",
            "dim_feedforward",
            "dropout",
            "positional_encoding_type",
            "max_sequence_length",
            "norm_first",
            "activation",
            "temporal_pooling",
        )
        missing = [field for field in required if field not in run_config]
        if missing:
            raise ValueError(
                "Transformer checkpoint run_config is missing fields: "
                + ", ".join(missing)
            )
        if run_config["architecture_version"] != "v1":
            raise ValueError("Transformer checkpoint architecture_version must be 'v1'")
        if run_config["positional_encoding_type"] != "sinusoidal":
            raise ValueError("Transformer checkpoint positional encoding must be sinusoidal")
        if run_config["temporal_pooling"] != "mean":
            raise ValueError("Transformer checkpoint temporal_pooling must be mean")
    architecture_contract = architecture_contract_from_config(run_config)
    saved_contract = payload.get("model_architecture_contract")
    if architecture == "ResNet18TemporalTransformer" and not isinstance(saved_contract, Mapping):
        raise ValueError("Transformer checkpoint is missing model_architecture_contract")
    if saved_contract is not None and (
        not isinstance(saved_contract, Mapping)
        or dict(saved_contract) != architecture_contract
    ):
        raise ValueError("checkpoint model_architecture_contract disagrees with run_config")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise ValueError("checkpoint is missing a valid best epoch")
    return {
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": epoch,
        "best_val_loss": float(best_val_loss),
        "architecture": architecture,
        "architecture_contract": architecture_contract,
        "run_id": checkpoint_sha256,
    }


def finalize_validation(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    val_loader: Any,
    output_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the best checkpoint and select a threshold using validation only."""

    _validate_loader_split(val_loader, "val", "validation")
    checkpoint_payload = load_checkpoint(checkpoint_path, model=model, map_location=device, restore_rng=False)
    provenance = _checkpoint_provenance(checkpoint_path, checkpoint_payload)
    if not isinstance(manifest_sha256, str) or not manifest_sha256:
        raise ValueError("runtime manifest_sha256 is required for validation finalization")
    if manifest_sha256 != provenance["manifest_sha256"]:
        raise ValueError("runtime manifest_sha256 does not match checkpoint provenance")
    result = evaluate_loader(model, val_loader, device=device)
    selection = select_threshold(result["labels"], result["probabilities"])
    metrics = compute_binary_metrics(
        result["labels"],
        result["probabilities"],
        selection["threshold"],
    )
    output = Path(output_dir)
    predictions_path = _write_predictions(output / "validation_predictions.csv", result, selection["threshold"])
    validation_predictions_sha256 = sha256_file(predictions_path)
    metrics_payload = dict(metrics)
    metrics_payload.update(
        {
            "evaluation_split": "derived_val",
            "selection_split": "validation",
            "evaluation_loss": float(result["loss"]),
            **provenance,
            "validation_predictions_sha256": validation_predictions_sha256,
            "class_counts": _class_counts(result["labels"]),
        }
    )
    metrics_path = _write_json(output / "validation_metrics.json", metrics_payload)
    selection.update(
        {
            "schema": THRESHOLD_ARTIFACT_SCHEMA,
            **provenance,
            "validation_predictions_sha256": validation_predictions_sha256,
            "threshold_rule": THRESHOLD_RULE,
        }
    )
    threshold_path = _write_json(output / "selected_threshold.json", selection)
    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "threshold": threshold_path,
        "threshold_value": selection["threshold"],
        "metrics_payload": metrics_payload,
    }


def evaluate_test_fixed_threshold(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    threshold_path: str | Path,
    test_loader: Any,
    output_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    manifest_sha256: str | None = None,
    run_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate derived test once using an existing validation threshold."""

    output = Path(output_dir)
    protected_outputs = (
        output / "test_predictions.csv",
        output / "test_metrics.json",
        output / "confusion_matrix.png",
    )
    existing = [path.name for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing final-test artifacts: " + ", ".join(existing)
        )
    threshold_artifact = _read_threshold_artifact(threshold_path)
    threshold = threshold_artifact["threshold"]
    _validate_loader_split(test_loader, "test", "test")
    checkpoint_payload = load_checkpoint(checkpoint_path, model=model, map_location=device, restore_rng=False)
    provenance = _checkpoint_provenance(checkpoint_path, checkpoint_payload)
    if not isinstance(manifest_sha256, str) or not manifest_sha256:
        raise ValueError("runtime manifest_sha256 is required for final test evaluation")
    if manifest_sha256 != provenance["manifest_sha256"]:
        raise ValueError("runtime manifest_sha256 does not match checkpoint provenance")
    for field in ("manifest_sha256", "checkpoint_sha256", "best_epoch", "run_id"):
        if threshold_artifact.get(field) != provenance[field]:
            raise ValueError(f"threshold artifact {field} does not match checkpoint provenance")
    for field in ("architecture", "architecture_contract"):
        if (
            field in threshold_artifact
            or provenance["architecture"] == "ResNet18TemporalTransformer"
        ) and threshold_artifact.get(field) != provenance[field]:
            raise ValueError(f"threshold artifact {field} does not match checkpoint provenance")
    validation_predictions_sha256 = threshold_artifact.get("validation_predictions_sha256")
    if (
        validation_predictions_sha256 is not None
        or provenance["architecture"] == "ResNet18TemporalTransformer"
    ):
        if (
            not isinstance(validation_predictions_sha256, str)
            or len(validation_predictions_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in validation_predictions_sha256.lower()
            )
        ):
            raise ValueError(
                "threshold artifact is missing a valid validation_predictions_sha256"
            )
        validation_predictions_path = Path(threshold_path).with_name(
            "validation_predictions.csv"
        )
        if not validation_predictions_path.is_file():
            raise ValueError(
                "validation_predictions.csv is required beside the threshold artifact"
            )
        if sha256_file(validation_predictions_path) != validation_predictions_sha256:
            raise ValueError(
                "threshold artifact validation_predictions_sha256 does not match "
                "validation_predictions.csv"
            )
    result = evaluate_loader(model, test_loader, device=device)
    metrics = compute_binary_metrics(result["labels"], result["probabilities"], threshold)
    predictions_path = _write_predictions(output / "test_predictions.csv", result, threshold)
    metrics_payload = dict(metrics)
    metrics_payload.update(
        {
            "evaluation_split": "derived_test",
            "threshold_source": "validation",
            "evaluation_loss": float(result["loss"]),
            "threshold_sha256": sha256_file(threshold_path),
            **provenance,
            "class_counts": _class_counts(result["labels"]),
        }
    )
    if run_context is not None:
        metrics_payload["run_context"] = dict(run_context)
    metrics_path = _write_json(output / "test_metrics.json", metrics_payload)
    confusion_path = _write_confusion_matrix(
        output / "confusion_matrix.png",
        metrics["confusion_matrix"],
        "RWF-2000 derived test confusion matrix",
    )
    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "confusion_matrix": confusion_path,
        "threshold_value": threshold,
        "metrics_payload": metrics_payload,
    }
