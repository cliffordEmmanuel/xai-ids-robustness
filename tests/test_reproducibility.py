"""Tests for reproducibility utilities."""

from __future__ import annotations

import json
import re

import numpy as np

from xai_ids.reproducibility import (
    capture_code_state,
    capture_versions,
    generate_run_id,
    make_run_dir,
    set_seed,
    write_provenance,
)

RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z_[0-9a-f]{8}$")


def test_set_seed_is_deterministic() -> None:
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_different_seeds_differ() -> None:
    set_seed(1)
    a = np.random.rand(5)
    set_seed(2)
    b = np.random.rand(5)
    assert not np.allclose(a, b)


def test_run_id_format_and_seed_sensitivity() -> None:
    rid = generate_run_id(seed=42)
    assert RUN_ID_RE.match(rid), rid
    # Different seeds should (almost surely) produce different hashes.
    assert generate_run_id(seed=1).split("_")[1] != generate_run_id(seed=2).split("_")[1]


def test_make_run_dir_creates(tmp_path) -> None:
    d = make_run_dir(tmp_path, "20260101T000000Z_deadbeef")
    assert d.exists() and d.is_dir()


def test_capture_versions_has_core() -> None:
    v = capture_versions()
    assert "python" in v
    assert "numpy" in v


def test_capture_code_state_has_a_stable_shape() -> None:
    state = capture_code_state()
    assert set(state) == {"git_revision", "worktree_dirty", "changed_file_count"}


def test_write_provenance(tmp_path) -> None:
    out = write_provenance(tmp_path, config=None, seed=7, extra={"stage": "test"})
    assert out.exists()
    record = json.loads(out.read_text())
    assert record["seed"] == 7
    assert "versions" in record and "python" in record["versions"]
    assert "code" in record
    assert record["extra"] == {"stage": "test"}
