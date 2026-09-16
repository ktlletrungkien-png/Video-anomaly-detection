"""PyTorch Dataset/DataLoader foundation for manifest-backed clips."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .decoder import OpenCVDecoder
from .manifest import ManifestRecord, load_manifest
from .preprocessing import PreprocessConfig, preprocess_frames
from .sampling import sample_uniform_indices


class RWF2000Dataset(Dataset[dict[str, Any]]):
    """Read one uniformly sampled 16-frame clip per manifest record."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest: str | Path | Sequence[ManifestRecord],
        *,
        split: str | None = None,
        decoder: Any | None = None,
        preprocess_config: PreprocessConfig | None = None,
        training: bool | None = None,
        seed: int = 42,
        num_frames: int = 16,
    ) -> None:
        self.root = Path(dataset_root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"RWF-2000 dataset root does not exist: {self.root}")
        self.records = list(load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest)
        if split is not None:
            self.records = [record for record in self.records if record.derived_split == split]
        self.decoder = decoder if decoder is not None else OpenCVDecoder()
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.training = (split == "train") if training is None else training
        self.seed = seed
        self.num_frames = num_frames
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")

    def __len__(self) -> int:
        return len(self.records)

    def _path_for(self, record: ManifestRecord) -> Path:
        root = self.root.resolve()
        path = (self.root / Path(record.relative_path)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Manifest path escapes dataset root: {record.relative_path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Video listed in manifest does not exist: {path}")
        return path

    def _frame_count(self, path: Path) -> int:
        for name in ("frame_count", "get_frame_count"):
            method = getattr(self.decoder, name, None)
            if method is not None:
                count = int(method(path))
                if count < 0:
                    raise RuntimeError(f"Decoder returned an invalid frame count for {path}: {count}")
                return count
        metadata = getattr(self.decoder, "metadata", None)
        if metadata is not None:
            info = metadata(path)
            if "frame_count" in info:
                return int(info["frame_count"])
        raise TypeError(
            "Injected decoder must provide frame_count(path) or get_frame_count(path) "
            "in addition to decode(path, indices)"
        )

    def _decode(self, path: Path, indices: Sequence[int]) -> Sequence[object]:
        method: Callable[..., Sequence[object]] | None = getattr(self.decoder, "decode", None)
        frames = method(path, indices) if method is not None else self.decoder(path, indices)
        if len(frames) != len(indices):
            raise RuntimeError(
                f"Decoder returned {len(frames)} frames for {len(indices)} requested from {path}"
            )
        return frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self._path_for(record)
        frame_count = self._frame_count(path)
        indices = sample_uniform_indices(frame_count, self.num_frames)
        frames = self._decode(path, indices)
        clip_rng = random.Random(self.seed + int(record.clip_id[:16], 16))
        tensor = preprocess_frames(
            frames,
            self.preprocess_config,
            training=self.training,
            rng=clip_rng,
        )
        return {
            "frames": tensor,
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "metadata": {
                "clip_id": record.clip_id,
                "relative_path": record.relative_path,
                "source_split": record.source_split,
                "derived_split": record.derived_split,
                "label": record.label,
                "label_name": record.label_name,
                "frame_indices": indices,
            },
        }


def seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy worker RNGs from PyTorch's worker seed."""

    del worker_id  # torch.initial_seed already incorporates the worker id.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


def make_dataloader(
    dataset: RWF2000Dataset,
    *,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    drop_last: bool = False,
) -> DataLoader:
    """Create a reproducibly seeded loader; tests can use ``num_workers=0``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
    )
