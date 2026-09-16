"""Injectable video decoders and the OpenCV default implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


class DecoderError(RuntimeError):
    """Raised when a video cannot be opened or decoded as requested."""


class OpenCVDecoder:
    """Sequential OpenCV decoder.

    Frames are read from index zero through the largest requested index.  This
    avoids frame reordering caused by random seeking, converts OpenCV's BGR
    output to RGB, and does not read beyond the requested clip.
    """

    def __init__(self, *, cv2_module: Any | None = None) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module  # type: ignore
            except ImportError as exc:
                raise DecoderError(
                    "OpenCV is required for real-video decoding; install the project dependencies "
                    "or inject a decoder for synthetic tests"
                ) from exc
        self.cv2 = cv2_module

    def frame_count(self, path: str | Path) -> int:
        capture = self.cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise DecoderError(f"Could not open video for frame count: {path}")
        try:
            count = int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if count < 0:
            raise DecoderError(f"Video reported an invalid frame count ({count}): {path}")
        return count

    def decode(self, path: str | Path, indices: Sequence[int]) -> list[Any]:
        requested = [int(index) for index in indices]
        if any(index < 0 for index in requested):
            raise DecoderError(f"Negative frame index requested for {path}")
        if not requested:
            return []
        if requested != sorted(requested) or len(set(requested)) != len(requested):
            raise DecoderError("Requested frame indices must be unique and monotonically increasing")

        capture = self.cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise DecoderError(f"Could not open video: {path}")
        output: list[Any] = []
        requested_position = 0
        frame_index = 0
        try:
            while requested_position < len(requested):
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise DecoderError(
                        f"Decode failed at frame {frame_index} while reading {path}; "
                        f"requested {requested[requested_position]}"
                    )
                if frame_index == requested[requested_position]:
                    output.append(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
                    requested_position += 1
                frame_index += 1
        finally:
            capture.release()
        if len(output) != len(requested):
            raise DecoderError(
                f"Decoded {len(output)} of {len(requested)} requested frames from {path}"
            )
        return output

    def __call__(self, path: str | Path, indices: Sequence[int]) -> list[Any]:
        return self.decode(path, indices)
