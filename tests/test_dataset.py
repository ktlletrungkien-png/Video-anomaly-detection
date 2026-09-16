from pathlib import Path

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
