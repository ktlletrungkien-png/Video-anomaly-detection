"""Small, testable training and checkpoint foundation for the RWF baseline.

This module intentionally stops at train/validation fitting. Metric selection,
thresholds, final test evaluation, and a CLI belong to later Phase 2 slices.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import numbers
from pathlib import Path
import platform
import random
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from .train_config import TrainConfig


def set_global_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and torch and return the applied policy."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be a boolean")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)
    return {
        "seed": seed,
        "deterministic": deterministic,
        "torch_deterministic_algorithms": deterministic,
        "cudnn_deterministic": deterministic,
        "cudnn_benchmark": not deterministic,
        "cuda_available_at_seed_time": cuda_available,
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture process RNG states in a torch-checkpoint-serializable mapping."""

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    state: dict[str, Any] = {
        # Use primitive containers for NumPy/Python states so the checkpoint
        # remains loadable by torch's safe weights-only reader as well.
        "python": [python_state[0], list(python_state[1]), python_state[2]],
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = []
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore states produced by :func:`capture_rng_state`."""

    python_state = state["python"]
    random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
    numpy_state = state["numpy"]
    if isinstance(numpy_state, Mapping):
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["state"], dtype=np.uint32),
                numpy_state["pos"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
    else:
        # Compatibility with checkpoints produced before primitive-state
        # serialization was introduced.
        np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch"])
    cuda_states = state.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def build_optimizer(
    model: nn.Module,
    config: TrainConfig | None = None,
    *,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
) -> Optimizer:
    """Build AdamW over trainable parameters only."""

    config = config or TrainConfig()
    lr = config.learning_rate if learning_rate is None else learning_rate
    decay = config.weight_decay if weight_decay is None else weight_decay
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("cannot build optimizer: model has no trainable parameters")
    return AdamW(trainable, lr=lr, weight_decay=decay)


def _device_and_dtype(model: nn.Module, device: torch.device | str) -> tuple[torch.device, torch.dtype]:
    target = torch.device(device)
    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise ValueError("model must have at least one parameter") from exc
    return target, parameter.dtype


def _batch_tensors(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("training batches must be mappings with 'frames' and 'label'")
    if "frames" not in batch or "label" not in batch:
        raise KeyError("training batch must contain 'frames' and 'label'")
    frames = batch["frames"]
    labels = batch["label"]
    if not isinstance(frames, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("batch 'frames' and 'label' values must be torch tensors")
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if frames.ndim < 1 or frames.shape[0] == 0:
        raise ValueError("batch frames must have a non-empty batch dimension")
    if labels.shape[0] != frames.shape[0]:
        raise ValueError(
            f"frames and labels have different batch sizes: {frames.shape[0]} vs {labels.shape[0]}"
        )
    return frames.to(device=device, dtype=dtype), labels.to(device=device, dtype=dtype)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    *,
    device: torch.device | str = "cpu",
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Train one epoch and return sample-weighted loss statistics."""

    model.train()
    target_device, model_dtype = _device_and_dtype(model, device)
    model.to(target_device)
    criterion = criterion or nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        frames, labels = _batch_tensors(batch, device=target_device, dtype=model_dtype)
        optimizer.zero_grad(set_to_none=True)
        logits = model(frames)
        if not isinstance(logits, Tensor) or logits.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError("model must return logits shaped [B]")
        loss = criterion(logits, labels)
        if loss.ndim != 0:
            raise ValueError("criterion must return a scalar loss")
        loss.backward()
        optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_samples += count
    if total_samples == 0:
        raise ValueError("cannot train on an empty loader")
    return {"loss": total_loss / total_samples, "num_samples": total_samples}


def _metadata_records(metadata: Any, batch_size: int) -> list[dict[str, Any]]:
    if metadata is None:
        return []
    if isinstance(metadata, list) and all(isinstance(item, Mapping) for item in metadata):
        return [
            {
                key: item[key]
                for key in ("clip_id", "relative_path")
                if key in item
            }
            for item in metadata
        ]
    if not isinstance(metadata, Mapping):
        raise TypeError("batch metadata must be a mapping or list of mappings")

    records: list[dict[str, Any]] = []
    for index in range(batch_size):
        record: dict[str, Any] = {}
        for key in ("clip_id", "relative_path"):
            if key not in metadata:
                continue
            value = metadata[key]
            if isinstance(value, Tensor):
                item = value[index].item() if value.ndim > 0 else value.item()
            elif isinstance(value, (list, tuple)):
                item = value[index]
            else:
                item = value
            record[key] = item
        records.append(record)
    return records


def evaluate_loader(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str = "cpu",
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate without gradients and collect labels, logits and metadata."""

    model.eval()
    target_device, model_dtype = _device_and_dtype(model, device)
    model.to(target_device)
    criterion = criterion or nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0
    all_labels: list[Tensor] = []
    all_logits: list[Tensor] = []
    all_probabilities: list[Tensor] = []
    all_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            frames, labels = _batch_tensors(batch, device=target_device, dtype=model_dtype)
            logits = model(frames)
            if not isinstance(logits, Tensor) or logits.ndim != 1 or logits.shape[0] != labels.shape[0]:
                raise ValueError("model must return logits shaped [B]")
            loss = criterion(logits, labels)
            if loss.ndim != 0:
                raise ValueError("criterion must return a scalar loss")
            count = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * count
            total_samples += count
            all_labels.append(labels.detach().cpu())
            all_logits.append(logits.detach().cpu())
            all_probabilities.append(torch.sigmoid(logits.detach()).cpu())
            all_records.extend(_metadata_records(batch.get("metadata"), count))
    if total_samples == 0:
        raise ValueError("cannot evaluate an empty loader")
    return {
        "loss": total_loss / total_samples,
        "num_samples": total_samples,
        "labels": torch.cat(all_labels),
        "logits": torch.cat(all_logits),
        "probabilities": torch.cat(all_probabilities),
        "records": all_records,
    }


def _records_for_loader(loader: Any) -> list[Any] | None:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    records = getattr(dataset, "records", None)
    if records is not None:
        return list(records)
    indices = getattr(dataset, "indices", None)
    parent = getattr(dataset, "dataset", None)
    parent_records = getattr(parent, "records", None)
    if indices is not None and parent_records is not None:
        return [parent_records[index] for index in indices]
    return None


def _validate_loader_split(
    loader: Any,
    expected: str,
    name: str,
    *,
    expected_source: str | None = None,
) -> list[Any]:
    """Fail closed unless a loader exposes the exact derived/source split."""

    records = _records_for_loader(loader)
    if records is None:
        raise ValueError(f"{name} loader must expose manifest records for split validation")
    if not records:
        raise ValueError(f"{name} loader has no records")
    expected_source = expected_source or {"train": "train", "val": "train", "test": "val"}.get(expected)
    if expected_source is None:
        raise ValueError(f"no source-split policy is defined for derived split {expected!r}")
    actual = {getattr(record, "derived_split", None) for record in records}
    if actual != {expected}:
        raise ValueError(
            f"{name} loader must contain only derived split {expected!r}, "
            f"got {sorted(actual, key=str)!r}"
        )
    source = {getattr(record, "source_split", None) for record in records}
    if source != {expected_source}:
        raise ValueError(
            f"{name} loader must contain source split {expected_source!r} "
            f"for derived split {expected!r}, got {sorted(source, key=str)!r}"
        )
    return records


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(dirty) if dirty is not None else None}


def _json_safe(value: Any, *, field: str = "metadata") -> Any:
    """Normalize public metadata to primitives accepted by safe torch loading."""

    if value is None or isinstance(value, (bool, int, float)):
        # Cast numeric subclasses (including NumPy scalars) to built-ins.
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        # torch.__version__ can be a TorchVersion subclass of str; casting is
        # necessary for weights_only=True checkpoint inspection.
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, field=f"{field}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, field=f"{field}[{index}]") for index, item in enumerate(value)]
    raise TypeError(
        f"{field} contains unsupported metadata type {type(value).__name__}; "
        "use JSON-safe primitives"
    )


def environment_metadata(config: TrainConfig, *, device: torch.device | str) -> dict[str, Any]:
    def distribution_version(name: str) -> str | None:
        try:
            return str(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            return None

    cuda_available = bool(torch.cuda.is_available())
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = None
    metadata = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision_version) if torchvision_version is not None else None,
        "numpy_version": distribution_version("numpy"),
        "scikit_learn_version": distribution_version("scikit-learn"),
        "opencv_version": distribution_version("opencv-python"),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "freeze_backbone": config.freeze_backbone,
        "weights": config.weights,
        "deterministic": config.deterministic,
    }
    metadata.update(_git_state())
    return metadata


def _config_for_run(config: TrainConfig, *, device: torch.device | str, manifest_sha256: str | None) -> dict[str, Any]:
    values = config.to_dict()
    values.update({"device": str(device), "manifest_sha256": manifest_sha256})
    return values


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_val_loss: float,
    run_config: Mapping[str, Any],
    manifest_sha256: str | None,
    environment: Mapping[str, Any],
) -> Path:
    """Save the versioned baseline checkpoint contract."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_run_config = _json_safe(run_config, field="run_config")
    safe_environment = _json_safe(environment, field="environment")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": float(best_val_loss),
        "selection_metric": "val_loss",
        "selection_mode": "min",
        "checkpoint_rule": "min_validation_loss_earliest_tie",
        "run_config": safe_run_config,
        "manifest_sha256": manifest_sha256,
        "environment": safe_environment,
        "rng_state": capture_rng_state(),
    }
    torch.save(payload, output)
    return output


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: torch.device | str = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint with strict model-state validation."""

    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.6 has no weights_only argument.
        payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng and "rng_state" in payload:
        restore_rng_state(payload["rng_state"])
    return dict(payload)


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    fields = ["epoch", "train_loss", "val_loss", "learning_rate"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in history)


def fit(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: TrainConfig,
    output_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    manifest_sha256: str | None = None,
    run_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit on derived train/validation loaders and save the best checkpoint."""

    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    _validate_loader_split(train_loader, "train", "train", expected_source="train")
    _validate_loader_split(val_loader, "val", "validation", expected_source="train")
    target_device = torch.device(device)
    seed_policy = set_global_seed(config.seed, deterministic=config.deterministic)
    model.to(target_device)
    optimizer = build_optimizer(model, config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_config = _config_for_run(config, device=target_device, manifest_sha256=manifest_sha256)
    if run_context is not None:
        run_config.update(_json_safe(run_context, field="run_context"))
    # Keep the architecture contract in every checkpoint, including callers
    # that use ``fit`` directly instead of the Phase 2 CLI.  The CLI supplies
    # the canonical values explicitly; these fallbacks make synthetic/local
    # callers self-describing without changing their model implementation.
    run_config.setdefault("architecture", type(model).__name__)
    run_config.setdefault("feature_dim", getattr(model, "feature_dim", None))
    environment = environment_metadata(config, device=target_device)
    environment["seed_policy"] = seed_policy
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    best_epoch: int | None = None
    for epoch in range(config.epochs):
        set_epoch = getattr(getattr(train_loader, "dataset", None), "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)
        train_stats = train_one_epoch(model, train_loader, optimizer, device=target_device)
        val_stats = evaluate_loader(model, val_loader, device=target_device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "learning_rate": learning_rate,
        }
        history.append(row)
        # Strict improvement means an equal validation loss keeps the earlier
        # checkpoint, making selection deterministic and explicitly tie-safe.
        if val_stats["loss"] < best_val_loss:
            best_val_loss = float(val_stats["loss"])
            best_epoch = epoch + 1
            save_checkpoint(
                output / "best_checkpoint.pth",
                model=model,
                optimizer=optimizer,
                epoch=best_epoch,
                best_val_loss=best_val_loss,
                run_config=run_config,
                manifest_sha256=manifest_sha256,
                environment=environment,
            )
    _write_history(output / "history.csv", history)
    if best_epoch is None:
        raise RuntimeError("no best checkpoint was produced")
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": output / "best_checkpoint.pth",
        "run_config": run_config,
        "environment": environment,
    }


def sha256_file(path: str | Path) -> str:
    """Compute a manifest hash for checkpoint provenance."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
