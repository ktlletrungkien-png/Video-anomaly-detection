"""ResNet18 frame features with temporal average pooling.

The module deliberately is not imported from :mod:`rwf2000`'s package root.
The Phase 1 inspection and manifest CLI can therefore remain usable without
eagerly importing torchvision.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def _make_resnet18_feature_extractor(weights: Any) -> nn.Module:
    """Build a torchvision ResNet18 with its classification head removed."""

    # Keep torchvision lazy: importing this module must not make the Phase 1
    # CLI depend on torchvision just to inspect a dataset or manifest.
    from torchvision.models import resnet18

    backbone = resnet18(weights=weights)
    return nn.Sequential(*list(backbone.children())[:-1])


class ResNet18TemporalAverage(nn.Module):
    """Binary clip classifier using per-frame ResNet18 features.

    Args:
        weights: A torchvision ResNet18 weights enum, ``"DEFAULT"`` for the
            ImageNet weights, or ``None`` for an uninitialized offline model.
        freeze_backbone: If true, freeze the feature extractor and keep it in
            evaluation mode whenever ``model.train()`` is called. This keeps
            frozen BatchNorm buffers stable while the classifier is trained.
        backbone: Optional injected feature extractor for tests or controlled
            experiments. It must return ``[N, D]`` or ``[N, D, 1, 1]``.
        feature_dim: Feature width. For an injected backbone this can be
            omitted when the backbone exposes ``feature_dim``; ResNet18 uses
            512 by default.

    ``forward`` expects ``[B, T, 3, H, W]`` and returns raw logits ``[B]``.
    Sigmoid is intentionally left to the metric/inference layer.
    """

    def __init__(
        self,
        *,
        weights: Any = "DEFAULT",
        freeze_backbone: bool = True,
        backbone: nn.Module | None = None,
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()

        if backbone is None:
            if weights == "DEFAULT":
                from torchvision.models import ResNet18_Weights

                weights = ResNet18_Weights.DEFAULT
            backbone = _make_resnet18_feature_extractor(weights)
            inferred_dim = 512
        else:
            # A supplied backbone is already constructed; weights are not
            # meaningful in that path and must not trigger a download.
            inferred_dim = getattr(backbone, "feature_dim", 512)

        if feature_dim is None:
            feature_dim = inferred_dim
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError(f"feature_dim must be a positive integer, got {feature_dim!r}")

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.freeze_backbone = bool(freeze_backbone)
        self.classifier = nn.Linear(feature_dim, 1)

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "ResNet18TemporalAverage":
        """Set mode while preserving eval mode for a frozen backbone."""

        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _frame_features(self, frames: Tensor) -> Tensor:
        batch_size, time_steps = frames.shape[:2]
        flattened = frames.reshape(batch_size * time_steps, 3, *frames.shape[-2:])
        features = self.backbone(flattened)
        if not isinstance(features, Tensor) or features.ndim < 2:
            raise ValueError("backbone must return a tensor shaped [N, D] or [N, D, ...]")
        if features.shape[0] != batch_size * time_steps:
            raise ValueError(
                "backbone changed the frame batch dimension: "
                f"expected {batch_size * time_steps}, got {features.shape[0]}"
            )
        features = features.flatten(start_dim=1)
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"backbone returned feature width {features.shape[1]}, "
                f"expected {self.feature_dim}"
            )
        return features.reshape(batch_size, time_steps, self.feature_dim)

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

        temporal_features = self._frame_features(frames)
        pooled = temporal_features.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


def build_resnet18_temporal_average(
    *,
    weights: Any = "DEFAULT",
    freeze_backbone: bool = True,
    backbone: nn.Module | None = None,
    feature_dim: int | None = None,
) -> ResNet18TemporalAverage:
    """Construct the baseline with an explicit, readable public factory."""

    return ResNet18TemporalAverage(
        weights=weights,
        freeze_backbone=freeze_backbone,
        backbone=backbone,
        feature_dim=feature_dim,
    )
