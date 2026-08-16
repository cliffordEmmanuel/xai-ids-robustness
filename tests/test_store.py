"""Tests for feature-index map and DataBundle persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xai_ids.datasets.base import RawData
from xai_ids.datasets.preprocess import preprocess
from xai_ids.datasets.store import load_bundle, save_bundle


def _bundle():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = (rng.random(120) < 0.4).astype(int)
    raw = RawData(X=X, y=y, feature_names=["a", "b", "c"])
    return preprocess(raw, seed=1, normalise="standard", balance="none", attack_subset_size=15)


def test_feature_index():
    b = _bundle()
    assert b.feature_index == {"a": 0, "b": 1, "c": 2}


def test_save_load_roundtrip(tmp_path):
    b = _bundle()
    save_bundle(b, tmp_path / "cic")
    r = load_bundle(tmp_path / "cic")

    assert np.array_equal(b.X_train, r.X_train)
    assert np.array_equal(b.X_test, r.X_test)
    assert np.array_equal(b.attack_indices, r.attack_indices)
    assert r.feature_names == b.feature_names
    # scaler survives and transforms identically
    x = np.zeros((1, 3))
    assert np.allclose(b.scaler.transform(x), r.scaler.transform(x))


def test_save_load_without_scaler(tmp_path):
    rng = np.random.default_rng(2)
    raw = RawData(
        X=pd.DataFrame(rng.normal(size=(50, 2)), columns=["x", "y"]),
        y=(rng.random(50) < 0.5).astype(int),
        feature_names=["x", "y"],
    )
    b = preprocess(raw, normalise="none", balance="none")
    save_bundle(b, tmp_path / "d")
    r = load_bundle(tmp_path / "d")
    assert r.scaler is None
    assert np.array_equal(b.X_train, r.X_train)
