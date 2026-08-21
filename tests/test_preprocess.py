"""Tests for the shared RawData -> DataBundle preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xai_ids.datasets.base import RawData
from xai_ids.datasets.preprocess import preprocess


def _raw(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = (rng.random(n) < 0.3).astype(int)  # ~30% positives (imbalanced)
    return RawData(X=X, y=y, feature_names=["a", "b", "c", "d"])


def test_split_scale_and_shapes():
    raw = _raw()
    b = preprocess(raw, seed=42, test_size=0.2, normalise="standard", balance="none")
    assert b.X_train.shape[1] == b.X_test.shape[1] == 4
    assert len(b.X_train) + len(b.X_test) == raw.n_samples
    # scaler fitted on train only -> train columns ~0 mean, ~1 std
    assert np.allclose(b.X_train.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(b.X_train.std(axis=0), 1, atol=1e-6)
    # test set is NOT forced to zero mean (scaler frozen from train)
    assert not np.allclose(b.X_test.mean(axis=0), 0, atol=1e-6)
    assert b.feature_names == ["a", "b", "c", "d"]


def test_frozen_and_deterministic():
    raw = _raw()
    b1 = preprocess(raw, seed=42, balance="none")
    b2 = preprocess(raw, seed=42, balance="none")
    assert np.array_equal(b1.X_test, b2.X_test)
    assert np.array_equal(b1.X_train, b2.X_train)


def test_undersample_balances_train_only():
    raw = _raw()
    b = preprocess(raw, seed=42, balance="undersample_train_only")
    _, counts = np.unique(b.y_train, return_counts=True)
    assert counts[0] == counts[1]  # train balanced
    # test remains imbalanced (untouched)
    _, tcounts = np.unique(b.y_test, return_counts=True)
    assert tcounts[0] != tcounts[1]


def test_no_scaler_when_none():
    raw = _raw()
    b = preprocess(raw, normalise="none", balance="none")
    assert b.scaler is None


def test_categorical_encoding_is_fitted_on_training_rows_only():
    # Test-only values must be represented by the fitted ``other`` bucket, not
    # become columns or affect the category vocabulary.
    train = pd.DataFrame(
        {
            "value": np.arange(20, dtype=float),
            "proto": ["tcp"] * 10 + ["udp"] * 10,
        }
    )
    test = pd.DataFrame(
        {
            "value": np.arange(20, 24, dtype=float),
            "proto": ["test_only"] * 4,
        }
    )
    from xai_ids.datasets.categorical import TrainOnlyCategoricalEncoder

    encoder = TrainOnlyCategoricalEncoder(columns=("proto",), top_n={"proto": 2})
    encoder.fit(train)
    transformed_test = encoder.transform(test)

    assert "proto_test_only" not in encoder.output_columns_
    assert transformed_test["proto_other"].eq(1).all()
