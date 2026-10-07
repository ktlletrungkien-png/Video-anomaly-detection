"""Models for the RWF-2000 violence-detection branch."""

from .resnet18_temporal_average import (
    ResNet18TemporalAverage,
    build_resnet18_temporal_average,
)
from .resnet18_temporal_transformer import (
    ResNet18TemporalTransformer,
    build_resnet18_temporal_transformer,
)


SUPPORTED_ARCHITECTURES = (
    "ResNet18TemporalAverage",
    "ResNet18TemporalTransformer",
)
_UNSET = object()

_CONTRACT_FIELDS = {
    "ResNet18TemporalAverage": (
        "architecture",
        "architecture_version",
        "feature_dim",
        "freeze_backbone",
    ),
    "ResNet18TemporalTransformer": (
        "architecture",
        "architecture_version",
        "feature_dim",
        "freeze_backbone",
        "num_layers",
        "nhead",
        "dim_feedforward",
        "dropout",
        "positional_encoding_type",
        "max_sequence_length",
        "norm_first",
        "activation",
        "temporal_pooling",
    ),
}


def architecture_contract_from_config(config):
    """Return the architecture identity that must survive checkpointing."""

    values = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    architecture = values.get("architecture", "ResNet18TemporalAverage")
    if architecture not in _CONTRACT_FIELDS:
        raise ValueError(
            f"unsupported architecture {architecture!r}; "
            f"expected one of {SUPPORTED_ARCHITECTURES!r}"
        )
    if architecture == "ResNet18TemporalAverage" and "architecture_version" not in values:
        # Phase 2 checkpoints created before Phase 3 did not record an
        # explicit version.  Their implementation is the v1 contract.
        values = {**values, "architecture_version": "v1"}
    missing = [field for field in _CONTRACT_FIELDS[architecture] if field not in values]
    if missing:
        raise ValueError(
            f"{architecture} config is missing architecture fields: {', '.join(missing)}"
        )
    return {field: values[field] for field in _CONTRACT_FIELDS[architecture]}


def model_architecture_contract(model):
    """Return a supported model's effective architecture contract, if any."""

    architecture = getattr(model, "architecture", None)
    if architecture is None:
        return None
    if architecture not in _CONTRACT_FIELDS:
        raise ValueError(f"model declares unsupported architecture {architecture!r}")
    missing = [
        field
        for field in _CONTRACT_FIELDS[architecture]
        if not hasattr(model, field)
    ]
    if missing:
        raise ValueError(
            f"model {architecture} is missing architecture attributes: {', '.join(missing)}"
        )
    return {field: getattr(model, field) for field in _CONTRACT_FIELDS[architecture]}


def validate_model_architecture_contract(model, config):
    """Reject a model whose effective architecture disagrees with config."""

    actual = model_architecture_contract(model)
    if actual is None:
        return None
    expected = architecture_contract_from_config(config)
    if actual != expected:
        mismatches = [
            f"{field}: model={actual.get(field)!r}, config={expected.get(field)!r}"
            for field in sorted(set(actual) | set(expected))
            if actual.get(field) != expected.get(field)
        ]
        raise ValueError("model architecture contract mismatch: " + "; ".join(mismatches))
    return actual


def build_model_from_config(config, *, weights=_UNSET, backbone=None):
    """Build one of the supported architectures from a config-like object.

    ``config`` may be a :class:`TrainConfig` or any mapping exposing the same
    architecture fields.  ``weights`` is an explicit override used during
    checkpoint evaluation to guarantee no network access.
    """

    values = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    architecture = values.get("architecture", "ResNet18TemporalAverage")
    architecture_version = values.get("architecture_version")
    if architecture == "ResNet18TemporalTransformer" and architecture_version != "v1":
        raise ValueError("ResNet18TemporalTransformer architecture_version must be 'v1'")
    if (
        architecture == "ResNet18TemporalAverage"
        and architecture_version not in (None, "v1")
    ):
        raise ValueError("ResNet18TemporalAverage architecture_version must be 'v1' when present")
    resolved_weights = values.get("weights") if weights is _UNSET else weights
    if architecture == "ResNet18TemporalAverage":
        return build_resnet18_temporal_average(
            weights=resolved_weights,
            freeze_backbone=values.get("freeze_backbone", True),
            backbone=backbone,
            feature_dim=values.get("feature_dim", 512),
        )
    if architecture == "ResNet18TemporalTransformer":
        return build_resnet18_temporal_transformer(
            weights=resolved_weights,
            freeze_backbone=values.get("freeze_backbone", True),
            backbone=backbone,
            feature_dim=values.get("feature_dim", 512),
            num_layers=values.get("num_layers", 2),
            nhead=values.get("nhead", 4),
            dim_feedforward=values.get("dim_feedforward", 1024),
            dropout=values.get("dropout", 0.1),
            positional_encoding_type=values.get("positional_encoding_type", "sinusoidal"),
            max_sequence_length=values.get("max_sequence_length", values.get("num_frames", 16)),
            norm_first=values.get("norm_first", True),
            activation=values.get("activation", "gelu"),
            temporal_pooling=values.get("temporal_pooling", "mean"),
        )
    raise ValueError(
        f"unsupported architecture {architecture!r}; "
        f"expected one of {SUPPORTED_ARCHITECTURES!r}"
    )

__all__ = [
    "ResNet18TemporalAverage",
    "build_resnet18_temporal_average",
    "ResNet18TemporalTransformer",
    "build_resnet18_temporal_transformer",
    "SUPPORTED_ARCHITECTURES",
    "architecture_contract_from_config",
    "build_model_from_config",
    "model_architecture_contract",
    "validate_model_architecture_contract",
]
