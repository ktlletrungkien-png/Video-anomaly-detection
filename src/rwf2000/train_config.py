"""Validated, serializable configuration for the local baseline foundation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainConfig:
    """Configuration shared by the baseline training and checkpoint code.

    Paths are intentionally absent. Dataset and output locations are runtime
    inputs so this object can be reused locally and on Kaggle without putting
    environment-specific paths in committed configuration.
    """

    seed: int = 42
    epochs: int = 1
    batch_size: int = 2
    num_workers: int = 0
    num_frames: int = 16
    height: int = 224
    width: int = 224
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    weights: str | None = "DEFAULT"
    freeze_backbone: bool = True
    deterministic: bool = True
    checkpoint_rule: str = "min_validation_loss_earliest_tie"
    purpose: str = "local_smoke_only"

    def __post_init__(self) -> None:
        for name in (
            "seed",
            "epochs",
            "batch_size",
            "num_workers",
            "num_frames",
            "height",
            "width",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.num_frames <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("num_frames, height and width must be positive")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.weights not in (None, "DEFAULT"):
            raise ValueError("weights must be None or the exact identifier 'DEFAULT'")
        if not isinstance(self.freeze_backbone, bool):
            raise ValueError("freeze_backbone must be a boolean")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a boolean")
        if self.checkpoint_rule != "min_validation_loss_earliest_tie":
            raise ValueError(
                "checkpoint_rule must be 'min_validation_loss_earliest_tie'"
            )
        if not isinstance(self.purpose, str) or not self.purpose:
            raise ValueError("purpose must be a non-empty string")

    @property
    def lr(self) -> float:
        """Short alias used by optimizer setup and interactive notebooks."""

        return self.learning_rate

    @property
    def workers(self) -> int:
        """Short alias for callers that refer to DataLoader workers."""

        return self.num_workers

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainConfig":
        """Construct and validate a config from a mapping."""

        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown training configuration fields: {sorted(unknown)}")
        return cls(**dict(values))

    def save_json(self, path: str | Path) -> Path:
        """Persist this config as deterministic, human-readable JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def load_json(cls, path: str | Path) -> "TrainConfig":
        """Load and validate a JSON config."""

        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("training configuration JSON must contain an object")
        return cls.from_dict(values)
