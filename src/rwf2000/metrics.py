"""Binary violence metrics and validation-only threshold selection."""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _as_1d_float(values: Any, name: str) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    try:
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_inputs(labels: Any, probabilities: Any, threshold: float) -> tuple[np.ndarray, np.ndarray, float]:
    target = _as_1d_float(labels, "labels")
    scores = _as_1d_float(probabilities, "probabilities")
    if target.shape != scores.shape:
        raise ValueError(
            f"labels and probabilities must have equal shapes, got {target.shape} and {scores.shape}"
        )
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("labels must be binary with Violence encoded as 1")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise ValueError("threshold must be a finite number in [0, 1]")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be a finite number in [0, 1]")
    return target.astype(np.int64), scores, threshold_value


def compute_binary_metrics(labels: Any, probabilities: Any, threshold: float) -> dict[str, Any]:
    """Compute fixed-protocol binary violence metrics.

    Confusion-matrix order is ``[[TN, FP], [FN, TP]]`` and the positive class
    is Violence (label 1). ROC-AUC is deliberately an error for one-class
    targets instead of returning an invented value.
    """

    target, scores, threshold_value = _validate_inputs(labels, probabilities, threshold)
    if np.unique(target).size < 2:
        raise ValueError("ROC-AUC requires both binary classes in labels")
    predicted = (scores >= threshold_value).astype(np.int64)
    tn = int(((target == 0) & (predicted == 0)).sum())
    fp = int(((target == 0) & (predicted == 1)).sum())
    fn = int(((target == 1) & (predicted == 0)).sum())
    tp = int(((target == 1) & (predicted == 1)).sum())
    support = int(target.size)
    accuracy = (tp + tn) / support
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold_value,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(target, scores)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support": support,
        "class_support": {"Non-violence": int((target == 0).sum()), "Violence": int((target == 1).sum())},
    }


def select_threshold(labels: Any, probabilities: Any) -> dict[str, Any]:
    """Select a threshold using validation labels only.

    The deterministic rule is maximize Violence F1, then Violence recall, then
    choose the smallest threshold for any remaining tie.
    """

    target, scores, _ = _validate_inputs(labels, probabilities, 0.0)
    if np.unique(target).size < 2:
        raise ValueError("threshold selection requires both binary classes in labels")
    candidates = sorted({0.0, 1.0, *[float(value) for value in scores]})
    evidence: list[dict[str, float]] = []
    for candidate in candidates:
        metrics = compute_binary_metrics(target, scores, candidate)
        evidence.append(
            {
                "threshold": candidate,
                "f1": metrics["f1"],
                "recall": metrics["recall"],
            }
        )
    selected = min(
        evidence,
        key=lambda item: (-item["f1"], -item["recall"], item["threshold"]),
    )
    return {
        "threshold": selected["threshold"],
        "selection_split": "validation",
        "rule": "maximize Violence F1; tie-break by Violence recall; remaining ties choose smallest threshold",
        "candidate_count": len(candidates),
        "candidates": evidence,
        "selected_f1": selected["f1"],
        "selected_recall": selected["recall"],
        "support": int(target.size),
    }
