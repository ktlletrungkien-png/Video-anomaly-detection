from pathlib import Path

import pytest
import torch

from rwf2000.dataset import RWF2000Dataset, make_dataloader
from rwf2000.manifest import build_manifest
from rwf2000.preprocessing import PreprocessConfig


class SyntheticDecoder:
    """Synthetic infrastructure fixture; it is not RWF-2000 evidence."""

    def __init__(self):
        self.requested = {}

    def frame_count(self, path):
        return 32

    def decode(self, path, indices):
        self.requested[path.name] = list(indices)
        return [torch.full((4, 5, 3), float(index), dtype=torch.uint8) for index in indices]


def test_dataloader_smoke_shape_dtype_labels_and_temporal_order(tmp_path):
    root = tmp_path / "dataset"
    for class_name in ("Fight", "NonFight"):
        class_dir = root / "train" / class_name
        class_dir.mkdir(parents=True)
        (class_dir / "clip.mp4").write_bytes(b"synthetic fixture")
    for class_name in ("Fight", "NonFight"):
        class_dir = root / "test" / class_name
        class_dir.mkdir(parents=True)
        (class_dir / "test_clip.mp4").write_bytes(b"synthetic fixture")

    manifest = tmp_path / "manifest.csv"
    build_manifest(root, manifest, val_fraction=0, seed=7)
    decoder = SyntheticDecoder()
    dataset = RWF2000Dataset(
        root,
        manifest,
        split="train",
        decoder=decoder,
        preprocess_config=PreprocessConfig(height=8, width=8),
        training=False,
    )
    batch = next(iter(make_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0, seed=7)))

    assert tuple(batch["frames"].shape) == (2, 16, 3, 8, 8)
    assert batch["frames"].dtype == torch.float32
    assert batch["label"].dtype == torch.float32
    assert batch["label"].tolist() == [1.0, 0.0]
    assert all(
        left < right
        for indices in decoder.requested.values()
        for left, right in zip(indices, indices[1:])
    )
    assert all(max(indices) < 32 for indices in decoder.requested.values())


def _epoch_fixture(tmp_path: Path):
    root = tmp_path / "dataset"
    for class_name in ("Fight", "NonFight"):
        class_dir = root / "train" / class_name
        class_dir.mkdir(parents=True)
        (class_dir / "clip.mp4").write_bytes(b"synthetic fixture")
        test_dir = root / "test" / class_name
        test_dir.mkdir(parents=True)
        (test_dir / "test_clip.mp4").write_bytes(b"synthetic fixture")
    manifest = tmp_path / "manifest.csv"
    build_manifest(root, manifest, val_fraction=0, seed=7)
    return root, manifest


def test_training_augmentation_is_reproducible_per_epoch_and_changes_across_epochs(tmp_path):
    root, manifest = _epoch_fixture(tmp_path)
    config = PreprocessConfig(
        height=8,
        width=8,
        brightness_delta=0.25,
        contrast_range=(0.5, 1.5),
        crop_scale=(0.5, 1.0),
    )

    first_decoder = SyntheticDecoder()
    first = RWF2000Dataset(
        root,
        manifest,
        split="train",
        decoder=first_decoder,
        preprocess_config=config,
        training=True,
        seed=42,
    )
    first.set_epoch(0)
    epoch_zero = first[0]["frames"]
    first.set_epoch(1)
    epoch_one = first[0]["frames"]

    second = RWF2000Dataset(
        root,
        manifest,
        split="train",
        decoder=SyntheticDecoder(),
        preprocess_config=config,
        training=True,
        seed=42,
    )
    second.set_epoch(1)

    assert torch.equal(epoch_one, second[0]["frames"])
    assert not torch.equal(epoch_zero, epoch_one)


def test_evaluation_is_invariant_to_epoch(tmp_path):
    root, manifest = _epoch_fixture(tmp_path)
    dataset = RWF2000Dataset(
        root,
        manifest,
        split="test",
        decoder=SyntheticDecoder(),
        preprocess_config=PreprocessConfig(height=8, width=8),
        training=False,
        seed=42,
    )

    dataset.set_epoch(0)
    first = dataset[0]["frames"]
    dataset.set_epoch(9)
    second = dataset[0]["frames"]

    assert torch.equal(first, second)


@pytest.mark.parametrize("invalid_epoch", [-1, 1.5, True, "1"])
def test_set_epoch_rejects_invalid_values(tmp_path, invalid_epoch):
    root, manifest = _epoch_fixture(tmp_path)
    dataset = RWF2000Dataset(
        root,
        manifest,
        split="train",
        decoder=SyntheticDecoder(),
        preprocess_config=PreprocessConfig(height=8, width=8),
        training=True,
    )

    with pytest.raises(ValueError, match="non-negative integer"):
        dataset.set_epoch(invalid_epoch)
