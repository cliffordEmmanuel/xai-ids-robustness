"""Tests for the CLI entry point (commands that don't need real data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from xai_ids.cli import main

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_validate_ok(capsys):
    rc = main(["validate", str(CONFIGS / "tamper_example.yaml")])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_list_datasets(capsys):
    rc = main(["list", "datasets"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cicids2017" in out and "unsw_nb15" in out


def test_list_all(capsys):
    main(["list"])
    out = capsys.readouterr().out
    for kind in ("datasets", "models", "explainers", "attacks", "metrics"):
        assert kind in out


def test_no_command_errors():
    with pytest.raises(SystemExit):
        main([])


def test_new_commands_parse():
    from xai_ids.cli import build_parser

    for cmd in (
        "run",
        "train",
        "explain",
        "gen-grid",
        "grid",
        "validate-results",
        "analyze-results",
    ):
        args = build_parser().parse_args([cmd, "x.yaml"])
        assert args.command == cmd
    args = build_parser().parse_args(["sweep", "a.yaml", "b.yaml"])
    assert args.configs == ["a.yaml", "b.yaml"]
    args = build_parser().parse_args(
        ["validate-results", "results/v2", "--spec", "a.yaml", "--spec", "b.yaml"]
    )
    assert args.specs == ["a.yaml", "b.yaml"]
