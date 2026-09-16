"""Exact RWF-2000 class-name handling."""

from __future__ import annotations


CANONICAL_LABEL_MAP = {"Non-violence": 0, "Violence": 1}

# Class names are matched as complete normalized strings.  In particular,
# this intentionally does not use substring matching (e.g. ``NonFight`` must
# not be interpreted as ``Fight``).
_ALIASES = {
    "nonfight": (0, "Non-violence"),
    "nonviolence": (0, "Non-violence"),
    "non-violence": (0, "Non-violence"),
    "fight": (1, "Violence"),
    "violence": (1, "Violence"),
}


class UnknownClassError(ValueError):
    """Raised when a class directory is not a supported exact alias."""


def _normalize_class_name(value: str) -> str:
    if not isinstance(value, str):
        raise UnknownClassError(f"Unsupported class name {value!r}; expected a directory name")
    return value.strip().casefold()


def label_from_class_name(value: str) -> int:
    """Return the binary label for an exact, case-normalized class alias."""

    normalized = _normalize_class_name(value)
    try:
        return _ALIASES[normalized][0]
    except KeyError as exc:
        supported = ", ".join(sorted(_ALIASES))
        raise UnknownClassError(
            f"Unknown RWF-2000 class directory {value!r}; supported exact aliases: {supported}"
        ) from exc


def canonical_label_name(value: str) -> str:
    """Return the canonical name (``Non-violence`` or ``Violence``)."""

    normalized = _normalize_class_name(value)
    try:
        return _ALIASES[normalized][1]
    except KeyError as exc:
        supported = ", ".join(sorted(_ALIASES))
        raise UnknownClassError(
            f"Unknown RWF-2000 class directory {value!r}; supported exact aliases: {supported}"
        ) from exc
