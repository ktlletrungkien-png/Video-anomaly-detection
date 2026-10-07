from dataclasses import asdict

import pytest

from rwf2000.__main__ import _parser
from rwf2000.phase2_cli import _preprocess_from_run_config
from rwf2000.preprocessing import PreprocessConfig


def test_cli_help_parser_lists_phase1_and_phase2_commands():
    parser = _parser()
    command_action = next(action for action in parser._actions if action.dest == "command")

    assert set(command_action.choices) == {"inspect", "manifest", "train", "evaluate"}
    assert "root" in parser.parse_args(
        ["train", "--root", "root", "--manifest", "manifest.csv", "--config", "config.json", "--output", "out"]
    ).__dict__


def test_evaluate_reconstructs_complete_preprocess_contract():
    preprocess = PreprocessConfig(height=32, width=48)
    run_config = {
        "height": 32,
        "width": 48,
        "preprocess": asdict(preprocess),
    }
    restored = _preprocess_from_run_config(run_config)
    assert restored == preprocess
    assert isinstance(restored.mean, tuple)
    assert isinstance(restored.contrast_range, tuple)


def test_evaluate_rejects_incomplete_or_mismatched_preprocess_contract():
    preprocess = asdict(PreprocessConfig(height=32, width=48))
    with pytest.raises(ValueError, match="missing a preprocess"):
        _preprocess_from_run_config({"height": 32, "width": 48})
    with pytest.raises(ValueError, match="unknown fields"):
        _preprocess_from_run_config(
            {"height": 32, "width": 48, "preprocess": {**preprocess, "unknown": 1}}
        )
    with pytest.raises(ValueError, match="disagrees"):
        _preprocess_from_run_config(
            {"height": 64, "width": 48, "preprocess": preprocess}
        )
