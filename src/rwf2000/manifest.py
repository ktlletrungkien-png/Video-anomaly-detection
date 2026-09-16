"""Deterministic clip-level manifests for an RWF-2000-style layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable, Sequence

from .labels import CANONICAL_LABEL_MAP, canonical_label_name, label_from_class_name


SCHEMA_VERSION = "rwf2000-manifest-v1"
DEFAULT_VIDEO_EXTENSIONS = frozenset({".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"})
MANIFEST_FIELDS = (
    "schema_version",
    "seed",
    "val_fraction",
    "label_map",
    "relative_path",
    "clip_id",
    "source_split",
    "derived_split",
    "label",
    "label_name",
)


class ManifestError(ValueError):
    """Raised when a dataset layout cannot produce a safe manifest."""


@dataclass(frozen=True)
class ManifestRecord:
    schema_version: str
    seed: int
    val_fraction: float
    label_map: str
    relative_path: str
    clip_id: str
    source_split: str
    derived_split: str
    label: int
    label_name: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "ManifestRecord":
        return cls(
            schema_version=row["schema_version"],
            seed=int(row["seed"]),
            val_fraction=float(row["val_fraction"]),
            label_map=row["label_map"],
            relative_path=row["relative_path"],
            clip_id=row["clip_id"],
            source_split=row["source_split"],
            derived_split=row["derived_split"],
            label=int(row["label"]),
            label_name=row["label_name"],
        )


def _stable_clip_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


def _iter_video_files(class_dir: Path, extensions: set[str]) -> Iterable[Path]:
    for path in sorted(class_dir.rglob("*"), key=lambda item: item.relative_to(class_dir).as_posix().casefold()):
        if path.is_file() and path.suffix.casefold() in extensions:
            yield path


def _discover_files(
    root: Path,
    source_splits: Sequence[str],
    extensions: set[str],
) -> list[tuple[str, Path, int, str]]:
    if not root.exists():
        raise FileNotFoundError(f"RWF-2000 dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"RWF-2000 dataset root is not a directory: {root}")
    if len(set(source_splits)) != len(source_splits):
        raise ManifestError(f"source split names must be unique, got {source_splits!r}")

    discovered: list[tuple[str, Path, int, str]] = []
    seen_relative: set[str] = set()
    seen_clip_keys: dict[str, str] = {}
    for source_split in source_splits:
        split_dir = root / source_split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Required source split directory is missing: {split_dir} "
                f"(pass the actual name explicitly if this dataset uses a different layout)"
            )
        class_dirs = sorted((path for path in split_dir.iterdir() if path.is_dir()), key=lambda p: p.name.casefold())
        for class_dir in class_dirs:
            label = label_from_class_name(class_dir.name)
            canonical_name = canonical_label_name(class_dir.name)
            for path in _iter_video_files(class_dir, extensions):
                relative_path = path.relative_to(root).as_posix()
                relative_key = relative_path.casefold()
                if relative_key in seen_relative:
                    raise ManifestError(f"Duplicate relative path discovered: {relative_path}")
                seen_relative.add(relative_key)

                # The path under the class directory is used as a conservative
                # overlap check across official train/test roots.  This catches
                # copied clips when no content hash or source metadata exists.
                source_relative = path.relative_to(class_dir).as_posix()
                overlap_key = f"{canonical_name.casefold()}/{source_relative.casefold()}"
                prior_split = seen_clip_keys.get(overlap_key)
                if prior_split is not None and prior_split != source_split:
                    raise ManifestError(
                        f"Potential clip overlap across source splits {prior_split!r} and "
                        f"{source_split!r}: {source_relative}"
                    )
                seen_clip_keys[overlap_key] = source_split
                discovered.append((source_split, path, label, canonical_name))
    return discovered


def _validation_count(class_size: int, val_fraction: float) -> int:
    if class_size < 2 or val_fraction == 0:
        return 0
    # Keep a non-empty train side and, for a non-zero fraction, a validation
    # example whenever a class has enough clips.  This is still clip-level and
    # deterministic; it avoids silently producing no validation split on tiny
    # synthetic fixtures.
    proposed = int(math.floor(class_size * val_fraction))
    return min(class_size - 1, max(1, proposed))


def build_manifest(
    dataset_root: str | Path,
    output_csv: str | Path,
    *,
    train_split: str = "train",
    test_split: str = "test",
    val_fraction: float = 0.2,
    seed: int = 42,
    video_extensions: Iterable[str] | None = None,
) -> list[ManifestRecord]:
    """Build and persist a deterministic train/validation/test manifest.

    Official ``test_split`` clips are copied unchanged to derived ``test``.
    Validation is derived only from official ``train_split`` at clip level;
    frame sequences and windows are never split.  ``output_csv`` is paired
    with deterministic JSON metadata at the same stem.
    """

    if not 0 <= val_fraction < 1:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if train_split == test_split:
        raise ManifestError("train_split and test_split must be different")
    extensions = {
        (extension if str(extension).startswith(".") else f".{extension}").casefold()
        for extension in (video_extensions or DEFAULT_VIDEO_EXTENSIONS)
    }
    if not extensions:
        raise ValueError("at least one video extension is required")
    root = Path(dataset_root)
    discovered = _discover_files(root, (train_split, test_split), extensions)

    train_by_label: dict[int, list[tuple[str, Path, int, str]]] = {0: [], 1: []}
    for item in discovered:
        source_split, path, label, canonical_name = item
        if source_split == train_split:
            train_by_label[label].append(item)

    rng = random.Random(seed)
    derived_by_relative: dict[str, str] = {}
    for label in (0, 1):
        clips = sorted(train_by_label[label], key=lambda item: item[1].relative_to(root).as_posix().casefold())
        rng.shuffle(clips)
        n_val = _validation_count(len(clips), val_fraction)
        for item in clips[:n_val]:
            derived_by_relative[item[1].relative_to(root).as_posix()] = "val"
        for item in clips[n_val:]:
            derived_by_relative[item[1].relative_to(root).as_posix()] = "train"

    assignments: list[tuple[str, Path, int, str, str]] = []
    for source_split, path, label, canonical_name in discovered:
        relative_path = path.relative_to(root).as_posix()
        derived_split = "test" if source_split == test_split else derived_by_relative[relative_path]
        assignments.append((source_split, path, label, canonical_name, derived_split))
    assignments.sort(key=lambda item: item[1].relative_to(root).as_posix().casefold())

    label_map = json.dumps(CANONICAL_LABEL_MAP, sort_keys=True, separators=(",", ":"))
    records = [
        ManifestRecord(
            schema_version=SCHEMA_VERSION,
            seed=seed,
            val_fraction=val_fraction,
            label_map=label_map,
            relative_path=path.relative_to(root).as_posix(),
            clip_id=_stable_clip_id(path.relative_to(root).as_posix()),
            source_split=source_split,
            derived_split=derived_split,
            label=label,
            label_name=canonical_name,
        )
        for source_split, path, label, canonical_name, derived_split in assignments
    ]

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    counts: dict[str, dict[str, int]] = {}
    for record in records:
        counts.setdefault(record.derived_split, {"Non-violence": 0, "Violence": 0})[
            record.label_name
        ] += 1
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "val_fraction": val_fraction,
        "label_map": CANONICAL_LABEL_MAP,
        "train_source_split": train_split,
        "test_source_split": test_split,
        "num_records": len(records),
        "counts_by_derived_split": counts,
        "limitations": [
            "Clip overlap is checked by relative path only; source-level leakage cannot be verified without dataset metadata.",
            "Counts describe only the supplied directory and do not verify the official RWF-2000 release; no model performance metrics are included.",
        ],
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return records


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    """Load and validate the CSV portion of a previously created manifest."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ManifestError(
                f"Manifest fields do not match {MANIFEST_FIELDS}: {reader.fieldnames}"
            )
        records = [ManifestRecord.from_row(row) for row in reader]
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for record in records:
        if record.clip_id in seen_ids or record.relative_path.casefold() in seen_paths:
            raise ManifestError("Manifest contains duplicate clip IDs or relative paths")
        seen_ids.add(record.clip_id)
        seen_paths.add(record.relative_path.casefold())
        if record.label_name != canonical_label_name(record.label_name):
            raise ManifestError(f"Non-canonical label name in manifest: {record.label_name!r}")
        if record.label != label_from_class_name(record.label_name):
            raise ManifestError(f"Label/name mismatch in manifest: {record.relative_path}")
    return records
