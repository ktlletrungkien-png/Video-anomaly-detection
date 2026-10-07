"""Models for the RWF-2000 violence-detection branch."""

from .resnet18_temporal_average import (
    ResNet18TemporalAverage,
    build_resnet18_temporal_average,
)

__all__ = ["ResNet18TemporalAverage", "build_resnet18_temporal_average"]
