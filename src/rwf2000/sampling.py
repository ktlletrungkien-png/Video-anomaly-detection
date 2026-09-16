"""Temporal clip sampling primitives."""

from __future__ import annotations


def sample_uniform_indices(
    num_frames: int,
    num_samples: int = 16,
    *,
    start: int = 0,
    stop: int | None = None,
) -> list[int]:
    """Sample ``num_samples`` ordered indices from ``[start, stop)``.

    The interval is half-open and clip-local: no index at or after ``stop``
    can be returned.  The first and last samples are the interval endpoints;
    integer arithmetic makes the result reproducible without floating-point
    rounding.  Phase 1 uses a strict short-clip policy and raises when the
    interval has fewer frames than requested.
    """

    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames < 0:
        raise ValueError(f"num_frames must be a non-negative integer, got {num_frames!r}")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        raise ValueError(f"num_samples must be a positive integer, got {num_samples!r}")
    if stop is None:
        stop = num_frames
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (start, stop)):
        raise ValueError("start and stop must be integers")
    if start < 0 or stop > num_frames or start >= stop:
        raise ValueError(
            f"sampling interval [{start}, {stop}) must lie within [0, {num_frames})"
        )
    span = stop - start
    if span < num_samples:
        raise ValueError(
            f"clip interval [{start}, {stop}) has {span} frames; "
            f"strict policy requires at least {num_samples}"
        )
    if num_samples == 1:
        return [start]
    last_offset = span - 1
    denominator = num_samples - 1
    return [start + (position * last_offset) // denominator for position in range(num_samples)]


# Descriptive alias for callers that prefer a verb-first name.
uniform_sample_indices = sample_uniform_indices
