"""RWF-2000 data-pipeline foundation.

This package only handles dataset inspection, reproducible split manifests,
clip sampling, decoding, preprocessing, and loading.  It does not contain a
model or training loop.
"""

from .decoder import DecoderError, OpenCVDecoder
from .labels import (
    CANONICAL_LABEL_MAP,
    UnknownClassError,
    canonical_label_name,
    label_from_class_name,
)
from .manifest import ManifestRecord, build_manifest, load_manifest
from .sampling import sample_uniform_indices, uniform_sample_indices

# Torch/OpenCV-backed components are intentionally not imported here so that
# manifest and sampling utilities remain inspectable before optional runtime
# dependencies are installed.

__all__ = [
    "CANONICAL_LABEL_MAP",
    "DecoderError",
    "ManifestRecord",
    "OpenCVDecoder",
    "UnknownClassError",
    "build_manifest",
    "canonical_label_name",
    "label_from_class_name",
    "load_manifest",
    "sample_uniform_indices",
    "uniform_sample_indices",
]
