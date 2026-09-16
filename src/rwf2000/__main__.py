"""Command line entry point: ``python -m rwf2000 inspect|manifest``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inspection import inspect_dataset
from .manifest import build_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RWF-2000 Phase 1 data-pipeline utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect an existing dataset root")
    inspect_parser.add_argument("--root", required=True, type=Path, help="dataset root (no download is performed)")
    inspect_parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="explicit split directory names (default: train test)",
    )
    inspect_parser.add_argument("--short-clip-length", type=int, default=16)
    inspect_parser.add_argument("--output", type=Path, help="optional JSON output path")

    manifest_parser = subparsers.add_parser("manifest", help="create a deterministic clip-level manifest")
    manifest_parser.add_argument("--root", required=True, type=Path, help="dataset root (no download is performed)")
    manifest_parser.add_argument("--output", required=True, type=Path, help="manifest CSV output path")
    manifest_parser.add_argument("--train-split", default="train", help="official training directory name")
    manifest_parser.add_argument("--test-split", default="test", help="official test directory name")
    manifest_parser.add_argument("--val-fraction", type=float, default=0.2)
    manifest_parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            report = inspect_dataset(
                args.root,
                splits=args.splits,
                short_clip_length=args.short_clip_length,
            )
            rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
        elif args.command == "manifest":
            records = build_manifest(
                args.root,
                args.output,
                train_split=args.train_split,
                test_split=args.test_split,
                val_fraction=args.val_fraction,
                seed=args.seed,
            )
            print(f"Wrote {len(records)} records to {args.output}")
            print(f"Wrote deterministic metadata to {args.output.with_suffix('.json')}")
    except (FileNotFoundError, NotADirectoryError, OSError, TypeError, ValueError) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
