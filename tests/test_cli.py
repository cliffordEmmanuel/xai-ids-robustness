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


def test_run_is_stubbed():
    with pytest.raises(SystemExit):
        main(["run", str(CONFIGS / "tamper_example.yaml")])


def test_no_command_errors():
    with pytest.raises(SystemExit):
        main([])
