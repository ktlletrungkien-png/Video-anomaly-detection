"""Layout and video metadata inspection without downloading data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from .labels import canonical_label_name, label_from_class_name
from .manifest import DEFAULT_VIDEO_EXTENSIONS


def _video_files(class_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        (
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in extensions
        ),
        key=lambda path: path.relative_to(class_dir).as_posix().casefold(),
    )


def _read_metadata(path: Path, cv2_module: Any, *, short_clip_length: int) -> dict[str, Any]:
    capture = cv2_module.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {
            "relative_path": path.name,
            "opened": False,
            "decode_error": "could not open video",
        }
    reported_count = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2_module.CAP_PROP_FPS))
    width = int(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT))
    decoded_count = 0
    decode_error: str | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None:
                decode_error = f"decoder returned an empty frame at index {decoded_count}"
                break
            decoded_count += 1
    finally:
        capture.release()
    if reported_count > 0 and decoded_count != reported_count and decode_error is None:
        decode_error = f"metadata frame count {reported_count} != decoded count {decoded_count}"
    duration = decoded_count / fps if fps > 0 else None
    return {
        "relative_path": path.name,
        "opened": True,
        "frame_count": reported_count,
        "decoded_frame_count": decoded_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "short_clip": decoded_count < short_clip_length,
        "decode_error": decode_error,
    }


def inspect_dataset(
    dataset_root: str | Path,
    *,
    splits: Sequence[str] = ("train", "test"),
    video_extensions: Iterable[str] | None = None,
    short_clip_length: int = 16,
) -> dict[str, Any]:
    """Inspect an explicit RWF-style root and return JSON-serializable data.

    The function reports what is discoverable from the supplied path only.  It
    does not infer a missing ``val`` directory, download data, or assert that a
    directory is the real RWF-2000 release.
    """

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"RWF-2000 dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"RWF-2000 dataset root is not a directory: {root}")
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("splits must be a non-empty sequence of unique names")
    if short_clip_length <= 0:
        raise ValueError("short_clip_length must be positive")
    extensions = {
        (extension if str(extension).startswith(".") else f".{extension}").casefold()
        for extension in (video_extensions or DEFAULT_VIDEO_EXTENSIONS)
    }
    report: dict[str, Any] = {
        "dataset_root": str(root),
        "requested_splits": list(splits),
        "short_clip_length": short_clip_length,
        "splits": {},
        "limitations": [
            "Inspection describes the supplied directory; it does not verify that the files are the official RWF-2000 release.",
            "Source-level duplicate/leakage verification is unavailable without dataset provenance metadata.",
        ],
    }
    try:
        import cv2
    except ImportError:
        cv2 = None
        report["limitations"].append(
            "OpenCV is unavailable, so frame count, FPS, dimensions, duration, and decode failures were not inspected."
        )

    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Requested split directory is missing: {split_dir}; pass its actual name explicitly"
            )
        info: dict[str, Any] = {
            "num_clips": 0,
            "counts_by_label": {"Non-violence": 0, "Violence": 0},
            "short_clips": 0,
            "decode_failures": [],
            "videos": [],
        }
        class_dirs = sorted((path for path in split_dir.iterdir() if path.is_dir()), key=lambda p: p.name.casefold())
        for class_dir in class_dirs:
            label = label_from_class_name(class_dir.name)
            canonical_name = canonical_label_name(class_dir.name)
            for path in _video_files(class_dir, extensions):
                relative_path = path.relative_to(root).as_posix()
                info["num_clips"] += 1
                info["counts_by_label"][canonical_name] += 1
                if cv2 is None:
                    metadata = {
                        "relative_path": relative_path,
                        "metadata_available": False,
                    }
                else:
                    metadata = _read_metadata(path, cv2, short_clip_length=short_clip_length)
                    metadata["relative_path"] = relative_path
                    if metadata.get("short_clip"):
                        info["short_clips"] += 1
                    if metadata.get("decode_error"):
                        info["decode_failures"].append(relative_path)
                info["videos"].append(
                    {
                        **metadata,
                        "label": label,
                        "label_name": canonical_name,
                    }
                )
        report["splits"][split] = info
    return report
