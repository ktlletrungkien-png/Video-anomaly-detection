"""Synthetic infrastructure fixtures, not evidence from the RWF-2000 release."""

from pathlib import Path

import pytest

from rwf2000.inspection import inspect_dataset


def _inspection_fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic-rwf"
    for split, class_name, count in (
        ("train", "Fight", 2),
        ("train", "NonFight", 3),
        ("test", "Violence", 1),
        ("test", "Non-violence", 4),
    ):
        class_dir = root / split / class_name
        class_dir.mkdir(parents=True)
        for index in range(count):
            (class_dir / f"clip_{index:02d}.mp4").write_bytes(b"synthetic placeholder")
    return root


def test_inspection_missing_root_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset root does not exist"):
        inspect_dataset(tmp_path / "missing")


def test_inspection_reports_deterministic_label_and_split_counts(tmp_path, monkeypatch):
    root = _inspection_fixture_root(tmp_path)
    # Keep this test independent of whether optional OpenCV is installed: the
    # layout/count path is what is under test, not real video decoding.
    monkeypatch.setattr("rwf2000.inspection._read_metadata", lambda *args, **kwargs: {"decode_error": None})
    report = inspect_dataset(root, splits=("train", "test"))

    assert list(report["splits"]) == ["train", "test"]
    assert report["splits"]["train"]["num_clips"] == 5
    assert report["splits"]["train"]["counts_by_label"] == {"Non-violence": 3, "Violence": 2}
    assert report["splits"]["test"]["num_clips"] == 5
    assert report["splits"]["test"]["counts_by_label"] == {"Non-violence": 4, "Violence": 1}
