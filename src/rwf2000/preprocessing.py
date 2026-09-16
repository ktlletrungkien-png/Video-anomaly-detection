"""Frame-to-tensor preprocessing shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PreprocessConfig:
    """Geometry, normalization, and optional train-only augmentation settings."""

    height: int = 224
    width: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    horizontal_flip_prob: float = 0.5
    brightness_delta: float = 0.1
    contrast_range: tuple[float, float] = (0.9, 1.1)
    crop_scale: tuple[float, float] = (0.9, 1.0)

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if len(self.mean) != 3 or len(self.std) != 3 or any(value <= 0 for value in self.std):
            raise ValueError("mean and std must each contain three valid channel values")
        if not 0 <= self.horizontal_flip_prob <= 1:
            raise ValueError("horizontal_flip_prob must be between 0 and 1")
        if self.brightness_delta < 0:
            raise ValueError("brightness_delta must be non-negative")
        low, high = self.contrast_range
        if low <= 0 or low > high:
            raise ValueError("contrast_range must contain positive low <= high")
        low, high = self.crop_scale
        if not 0 < low <= high <= 1:
            raise ValueError("crop_scale must satisfy 0 < low <= high <= 1")


def _as_frame_tensor(frames: Sequence[object] | torch.Tensor) -> torch.Tensor:
    """Convert RGB frames from ``T,H,W,C`` to float tensor ``T,C,H,W``."""

    if isinstance(frames, torch.Tensor):
        tensor = frames
        if tensor.ndim != 4:
            raise ValueError(f"expected 4-D frames tensor, got shape {tuple(tensor.shape)}")
        if tensor.shape[1] == 3:
            tensor = tensor
        elif tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError("frames tensor must have three RGB channels")
    else:
        try:
            frame_count = len(frames)
        except TypeError as exc:
            raise ValueError("frames must be a sequence of RGB arrays or tensors") from exc
        if frame_count == 0:
            raise ValueError("cannot preprocess an empty frame sequence")
        try:
            tensor = torch.stack([torch.as_tensor(frame) for frame in frames])
        except (TypeError, RuntimeError) as exc:
            raise ValueError("frames must be stackable RGB arrays or tensors") from exc
        if tensor.ndim != 4 or tensor.shape[-1] != 3:
            raise ValueError(f"expected frames shaped [T,H,W,3], got shape {tuple(tensor.shape)}")
        tensor = tensor.permute(0, 3, 1, 2)

    if tensor.shape[0] == 0 or tensor.shape[1] != 3 or tensor.shape[2] == 0 or tensor.shape[3] == 0:
        raise ValueError(f"expected non-empty RGB frames, got shape {tuple(tensor.shape)}")
    integer_input = not tensor.dtype.is_floating_point
    tensor = tensor.to(dtype=torch.float32)
    # uint8 input becomes [0, 1].  Floating input is accepted as either byte
    # scale or an already scaled image, but values outside byte range are an
    # error rather than silently changing the data.
    if torch.isfinite(tensor).logical_not().any() or tensor.min() < 0 or tensor.max() > 255:
        raise ValueError("frame values must be finite and in the range [0, 255]")
    if integer_input or tensor.max() > 1:
        tensor = tensor / 255.0
    return tensor


def preprocess_frames(
    frames: Sequence[object] | torch.Tensor,
    config: PreprocessConfig | None = None,
    *,
    training: bool = False,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Return normalized ``float32 [T, C, H, W]`` RGB frames.

    Train-only transforms draw their parameters once and apply them to the
    complete clip.  Evaluation and inference are deterministic.  The base
    geometry is a direct resize, intentionally shared between train and eval.
    No temporal transform is performed.
    """

    config = config or PreprocessConfig()
    tensor = _as_frame_tensor(frames)
    rng = rng or random.Random()

    if training:
        if rng.random() < config.horizontal_flip_prob:
            tensor = tensor.flip(-1)
        if config.brightness_delta:
            brightness = rng.uniform(-config.brightness_delta, config.brightness_delta)
            tensor = tensor + brightness
        low_contrast, high_contrast = config.contrast_range
        contrast = rng.uniform(low_contrast, high_contrast)
        tensor = tensor * contrast

        low_crop, high_crop = config.crop_scale
        crop_scale = rng.uniform(low_crop, high_crop)
        crop_height = max(1, int(round(tensor.shape[2] * crop_scale)))
        crop_width = max(1, int(round(tensor.shape[3] * crop_scale)))
        top = rng.randint(0, tensor.shape[2] - crop_height) if crop_height < tensor.shape[2] else 0
        left = rng.randint(0, tensor.shape[3] - crop_width) if crop_width < tensor.shape[3] else 0
        tensor = tensor[:, :, top : top + crop_height, left : left + crop_width]

    tensor = tensor.clamp(0.0, 1.0)
    tensor = F.interpolate(
        tensor,
        size=(config.height, config.width),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor(config.mean, dtype=torch.float32, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(config.std, dtype=torch.float32, device=tensor.device).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(dtype=torch.float32)
