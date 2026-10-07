"""ResNet18 frame features followed by a temporal Transformer encoder."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .resnet18_temporal_average import ResNet18TemporalAverage


class ResNet18TemporalTransformer(ResNet18TemporalAverage):
    """Binary clip classifier using frozen ResNet18 features and attention.

    The frame feature extractor is shared with the Phase 2 baseline.  A
    sinusoidal positional encoding is added before a batch-first Transformer
    encoder; the encoded sequence is mean pooled and mapped to one raw logit.
    """

    architecture = "ResNet18TemporalTransformer"
    architecture_version = "v1"

    def __init__(
        self,
        *,
        weights: Any = "DEFAULT",
        freeze_backbone: bool = True,
        backbone: nn.Module | None = None,
        feature_dim: int | None = None,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        positional_encoding_type: str = "sinusoidal",
        max_sequence_length: int = 16,
        norm_first: bool = True,
        activation: str = "gelu",
        temporal_pooling: str = "mean",
    ) -> None:
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if not isinstance(nhead, int) or isinstance(nhead, bool) or nhead <= 0:
            raise ValueError("nhead must be a positive integer")
        if not isinstance(dim_feedforward, int) or isinstance(dim_feedforward, bool) or dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be a finite number in [0, 1)")
        if positional_encoding_type != "sinusoidal":
            raise ValueError("positional_encoding_type must be 'sinusoidal'")
        if not isinstance(max_sequence_length, int) or isinstance(max_sequence_length, bool) or max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be a positive integer")
        if not isinstance(norm_first, bool):
            raise ValueError("norm_first must be a boolean")
        if activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        if temporal_pooling != "mean":
            raise ValueError("temporal_pooling must be 'mean'")

        # Transformer multi-head attention requires an even divisibility
        # contract; fail early rather than at a less actionable constructor
        # error deep inside torch.nn.
        resolved_feature_dim = feature_dim
        if resolved_feature_dim is not None and resolved_feature_dim % nhead != 0:
            raise ValueError("feature_dim must be divisible by nhead")

        super().__init__(
            weights=weights,
            freeze_backbone=freeze_backbone,
            backbone=backbone,
            feature_dim=feature_dim,
        )
        if self.feature_dim % nhead != 0:
            raise ValueError("feature_dim must be divisible by nhead")

        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = float(dropout)
        self.positional_encoding_type = positional_encoding_type
        self.max_sequence_length = max_sequence_length
        self.norm_first = norm_first
        self.activation = activation
        self.temporal_pooling = temporal_pooling

        layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.register_buffer(
            "positional_encoding",
            self._make_sinusoidal_encoding(max_sequence_length, self.feature_dim),
            persistent=True,
        )

    @staticmethod
    def _make_sinusoidal_encoding(length: int, feature_dim: int) -> Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        indices = torch.arange(0, feature_dim, 2, dtype=torch.float32)
        divisor = torch.exp(indices * (-math.log(10000.0) / feature_dim))
        encoding = torch.zeros(length, feature_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        # For odd feature widths, the cosine slice is shorter than the sine
        # slice.  ResNet18 uses an even width, but keeping this helper robust
        # makes injected test backbones easier to use.
        encoding[:, 1::2] = torch.cos(positions * divisor[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def forward(self, frames: Tensor) -> Tensor:
        """Return one raw violence logit per input clip."""

        if not isinstance(frames, Tensor):
            raise TypeError(f"frames must be a torch.Tensor, got {type(frames).__name__}")
        if frames.ndim != 5:
            raise ValueError(
                f"expected frames shaped [B, T, 3, H, W], got shape {tuple(frames.shape)}"
            )
        batch_size, time_steps, channels, height, width = frames.shape
        if batch_size == 0 or time_steps == 0 or height == 0 or width == 0:
            raise ValueError(
                "frames must have non-empty B, T, H and W dimensions; "
                f"got shape {tuple(frames.shape)}"
            )
        if channels != 3:
            raise ValueError(f"frames must have 3 RGB channels, got {channels}")
        if time_steps > self.max_sequence_length:
            raise ValueError(
                f"sequence length {time_steps} exceeds max_sequence_length "
                f"{self.max_sequence_length}"
            )

        features = self._frame_features(frames)
        encoded = features + self.positional_encoding[:, :time_steps].to(
            device=features.device, dtype=features.dtype
        )
        encoded = self.transformer(encoded)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


def build_resnet18_temporal_transformer(
    *,
    weights: Any = "DEFAULT",
    freeze_backbone: bool = True,
    backbone: nn.Module | None = None,
    feature_dim: int | None = None,
    num_layers: int = 2,
    nhead: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    positional_encoding_type: str = "sinusoidal",
    max_sequence_length: int = 16,
    norm_first: bool = True,
    activation: str = "gelu",
    temporal_pooling: str = "mean",
) -> ResNet18TemporalTransformer:
    """Construct the Phase 3 model with explicit architecture parameters."""

    return ResNet18TemporalTransformer(
        weights=weights,
        freeze_backbone=freeze_backbone,
        backbone=backbone,
        feature_dim=feature_dim,
        num_layers=num_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        positional_encoding_type=positional_encoding_type,
        max_sequence_length=max_sequence_length,
        norm_first=norm_first,
        activation=activation,
        temporal_pooling=temporal_pooling,
    )
