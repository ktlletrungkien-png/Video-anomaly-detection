from pathlib import Path

import pytest

from rwf2000.manifest import ManifestError, build_manifest


def _fixture_root(tmp_path: Path) -> Path:
    for split, classes, count in (
        ("train", ("Fight", "NonFight"), 4),
        ("test", ("Violence", "Non-violence"), 2),
    ):
        for class_name in classes:
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            for index in range(count):
                prefix = "test_" if split == "test" else "clip_"
                (class_dir / f"{prefix}{index:02d}.mp4").write_bytes(b"synthetic fixture")
    return tmp_path


def test_manifest_is_byte_identical_and_keeps_official_test(tmp_path):
    root = _fixture_root(tmp_path / "dataset")
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    first = build_manifest(root, first_csv, val_fraction=0.5, seed=123)
    second = build_manifest(root, second_csv, val_fraction=0.5, seed=123)

    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first_csv.with_suffix(".json").read_bytes() == second_csv.with_suffix(".json").read_bytes()
    assert [record.clip_id for record in first] == [record.clip_id for record in second]

    train_paths = {record.relative_path for record in first if record.derived_split == "train"}
    val_paths = {record.relative_path for record in first if record.derived_split == "val"}
    test_records = [record for record in first if record.source_split == "test"]
    assert train_paths.isdisjoint(val_paths)
    assert test_records
    assert all(record.derived_split == "test" for record in test_records)
    assert all(record.source_split == "test" for record in test_records)


def test_manifest_rejects_same_clip_name_across_source_splits(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train", "test"):
        path = root / split / "Fight"
        path.mkdir(parents=True)
        (path / "same.mp4").write_bytes(b"synthetic fixture")
    with pytest.raises(ManifestError, match="overlap"):
        build_manifest(root, tmp_path / "manifest.csv")
