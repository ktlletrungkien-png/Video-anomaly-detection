"""Lazy Phase 2 CLI orchestration for train and fixed-threshold evaluate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any


def add_phase2_parsers(subparsers: Any) -> None:
    """Register Phase 2 arguments without importing torch or model code."""

    train_parser = subparsers.add_parser("train", help="train the ResNet18 temporal-average baseline")
    _add_common_runtime_arguments(train_parser)
    train_parser.add_argument("--config", required=True, type=Path, help="validated training config JSON")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="evaluate derived test using an existing validation-selected threshold",
    )
    _add_common_runtime_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument("--threshold", required=True, type=Path)


def _add_common_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True, type=Path, help="dataset root")
    parser.add_argument("--manifest", required=True, type=Path, help="manifest CSV")
    parser.add_argument("--output", required=True, type=Path, help="run output directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def resolve_device(requested: str):
    """Resolve cpu/cuda/auto and report an actionable CUDA error."""

    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment")
    return torch.device(requested)


def _load_checkpoint_run_config(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("run_config"), dict):
        raise ValueError("checkpoint does not contain a run_config mapping")
    return payload["run_config"]


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    """Run train + validation threshold finalization; never load derived test."""

    from .dataset import RWF2000Dataset, make_dataloader
    from .evaluate_baseline import finalize_validation
    from .models import build_model_from_config
    from .preprocessing import PreprocessConfig
    from .train_baseline import fit, set_global_seed, sha256_file
    from .train_config import TrainConfig

    config = TrainConfig.load_json(args.config)
    device = resolve_device(args.device)
    root = args.root.resolve()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    manifest_hash = sha256_file(manifest)
    if (
        config.expected_manifest_sha256 is not None
        and manifest_hash != config.expected_manifest_sha256
    ):
        raise ValueError(
            "runtime manifest_sha256 does not match config expected_manifest_sha256"
        )
    # Initialization must be seeded before constructing the model.
    set_global_seed(config.seed, deterministic=config.deterministic)
    model = build_model_from_config(config)
    preprocess = PreprocessConfig(height=config.height, width=config.width)
    train_dataset = RWF2000Dataset(
        root,
        manifest,
        split="train",
        preprocess_config=preprocess,
        training=True,
        seed=config.seed,
        num_frames=config.num_frames,
    )
    val_dataset = RWF2000Dataset(
        root,
        manifest,
        split="val",
        preprocess_config=preprocess,
        training=False,
        seed=config.seed,
        num_frames=config.num_frames,
    )
    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    context = {
        "dataset_root": str(root),
        "manifest_path": str(manifest),
        "architecture": config.architecture,
        "architecture_version": config.architecture_version,
        "feature_dim": int(getattr(model, "feature_dim", 512)),
        "loss": "BCEWithLogitsLoss",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        "threshold_rule": "maximize Violence F1; tie-break by Violence recall; remaining ties choose smallest threshold",
        "preprocess": asdict(preprocess),
        "class_counts": {
            "train": _class_counts(train_dataset),
            "val": _class_counts(val_dataset),
        },
        "source_split_semantics": {
            "derived_train": "source train",
            "derived_val": "source train",
            "derived_test": "source val held out",
        },
    }
    fit_result = fit(
        model,
        train_loader,
        val_loader,
        config,
        output,
        device=device,
        manifest_sha256=manifest_hash,
        run_context=context,
    )
    validation_result = finalize_validation(
        model,
        fit_result["checkpoint"],
        val_loader,
        output,
        device=device,
        manifest_sha256=manifest_hash,
    )
    result = {
        "checkpoint": str(fit_result["checkpoint"]),
        "best_epoch": fit_result["best_epoch"],
        "validation_metrics": str(validation_result["metrics"]),
        "selected_threshold": str(validation_result["threshold"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _class_counts(dataset: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in dataset.records:
        key = str(record.label_name)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _preprocess_from_run_config(run_config: dict[str, Any]):
    """Reconstruct the complete preprocessing contract saved with a run.

    Evaluation must use the exact geometry, normalization and deterministic
    inference settings used by training.  Defaults are intentionally not
    accepted here: an old or hand-written checkpoint without the complete
    contract should fail before any test records are loaded.
    """

    from .preprocessing import PreprocessConfig

    raw = run_config.get("preprocess")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint run_config is missing a preprocess mapping")
    allowed = {field.name for field in fields(PreprocessConfig)}
    unknown = sorted(set(raw) - allowed)
    missing = sorted(allowed - set(raw))
    if unknown:
        raise ValueError(f"checkpoint preprocess contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"checkpoint preprocess is missing fields: {', '.join(missing)}")
    values = dict(raw)
    for field_name in ("mean", "std", "contrast_range", "crop_scale"):
        value = values[field_name]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"checkpoint preprocess field {field_name!r} must be a list or tuple")
        values[field_name] = tuple(value)
    for field_name in ("height", "width"):
        top_level = run_config.get(field_name)
        if top_level is None:
            raise ValueError(f"checkpoint run_config is missing top-level {field_name}")
        if int(top_level) != int(values[field_name]):
            raise ValueError(
                f"checkpoint {field_name} disagrees between run_config and preprocess"
            )
    return PreprocessConfig(**values)


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate only derived test with a pre-existing validation threshold."""

    from .dataset import RWF2000Dataset, make_dataloader
    from .evaluate_baseline import evaluate_test_fixed_threshold
    from .models import build_model_from_config
    from .train_baseline import sha256_file

    device = resolve_device(args.device)
    root = args.root.resolve()
    manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    threshold = args.threshold.resolve()
    output = args.output.resolve()
    manifest_sha256 = sha256_file(manifest)
    run_config = _load_checkpoint_run_config(checkpoint)
    expected_manifest = run_config.get("expected_manifest_sha256")
    if expected_manifest is not None and manifest_sha256 != expected_manifest:
        raise ValueError(
            "runtime manifest_sha256 does not match checkpoint expected_manifest_sha256"
        )
    num_frames = int(run_config.get("num_frames", 16))
    preprocess = _preprocess_from_run_config(run_config)
    batch_size = int(run_config.get("batch_size", 4))
    num_workers = int(run_config.get("num_workers", 0))
    test_dataset = RWF2000Dataset(
        root,
        manifest,
        split="test",
        preprocess_config=preprocess,
        training=False,
        seed=int(run_config.get("seed", 42)),
        num_frames=num_frames,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=int(run_config.get("seed", 42)),
    )
    # Never request pretrained weights during evaluation; checkpoint loading
    # supplies the trained state and avoids any network dependency.
    # Reconstruct the exact architecture recorded by the checkpoint.  The
    # explicit weights override is None so evaluation never downloads a
    # pretrained model; the checkpoint supplies all learned parameters.
    model = build_model_from_config(run_config, weights=None)
    result = evaluate_test_fixed_threshold(
        model,
        checkpoint,
        threshold,
        test_loader,
        output,
        device=device,
        manifest_sha256=manifest_sha256,
        run_context={
            "dataset_root": str(root),
            "manifest_path": str(manifest),
            "checkpoint_path": str(checkpoint),
            "threshold_path": str(threshold),
            "source_split_semantics": {"derived_test": "source val held out"},
        },
    )
    rendered = {
        "test_metrics": str(result["metrics"]),
        "test_predictions": str(result["predictions"]),
        "confusion_matrix": str(result["confusion_matrix"]),
        "threshold": result["threshold_value"],
    }
    print(json.dumps(rendered, indent=2, sort_keys=True))
    return rendered


def run_phase2(args: argparse.Namespace) -> int:
    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        raise ValueError(f"unsupported Phase 2 command: {args.command}")
    return 0
