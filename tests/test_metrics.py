import numpy as np
import pytest

from rwf2000.metrics import compute_binary_metrics, select_threshold


def test_binary_metrics_uses_fixed_confusion_order_and_support():
    result = compute_binary_metrics(
        labels=[0, 0, 1, 1],
        probabilities=[0.1, 0.8, 0.2, 0.9],
        threshold=0.5,
    )

    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["support"] == 4
    assert result["class_support"] == {"Non-violence": 2, "Violence": 2}
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_metrics_zero_division_is_zero_when_no_positive_is_predicted():
    result = compute_binary_metrics([0, 1], [0.1, 0.2], threshold=0.5)

    assert result["confusion_matrix"] == [[1, 0], [1, 0]]
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


@pytest.mark.parametrize(
    "labels, probabilities, threshold, match",
    [
        ([0], [0.5], 0.5, "ROC-AUC requires"),
        ([0, 1], [0.1], 0.5, "equal shapes"),
        ([[0, 1]], [[0.1, 0.9]], 0.5, "one-dimensional"),
        ([0, 1], [float("nan"), 0.9], 0.5, "finite"),
        ([0, 2], [0.1, 0.9], 0.5, "binary"),
        ([0, 1], [-0.1, 0.9], 0.5, "probabilities"),
        ([0, 1], [0.1, 0.9], 1.1, "threshold"),
    ],
)
def test_metrics_validate_inputs(labels, probabilities, threshold, match):
    with pytest.raises(ValueError, match=match):
        compute_binary_metrics(labels, probabilities, threshold)


def test_metrics_reject_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        compute_binary_metrics([], [], 0.5)


def test_threshold_selection_uses_f1_then_recall_then_smallest_threshold():
    result = select_threshold(
        labels=[0, 0, 1, 1],
        probabilities=[0.2, 0.4, 0.6, 0.8],
    )

    assert result["selection_split"] == "validation"
    assert result["threshold"] == 0.6
    assert result["rule"].startswith("maximize Violence F1")
    assert result["candidate_count"] == 6
    assert result["support"] == 4


def test_threshold_selection_is_deterministic_for_remaining_ties():
    first = select_threshold([0, 1], [0.25, 0.75])
    second = select_threshold(np.array([0, 1]), np.array([0.25, 0.75]))

    assert first == second
