"""Tests for the IDS models. Heavy-dep tests skip if torch/xgboost are absent."""

from __future__ import annotations

import numpy as np
import pytest

import xai_ids.models  # noqa: F401 - registers the models
from xai_ids.registry import MODELS

_DEP = {"xgboost": "xgboost", "mlp": "torch"}


def _require(module: str) -> None:
    """Skip if a heavy dep is missing or unusable (e.g. torch without CUDA libs)."""
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001 - any import failure => skip
        pytest.skip(f"{module} unavailable: {exc}")


def _data(n=240, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_models_registered():
    # Registration is lazy (no torch/xgboost needed to import the package).
    assert "xgboost" in MODELS
    assert "mlp" in MODELS


@pytest.mark.parametrize("name", ["xgboost", "mlp"])
def test_fit_predict_evaluate_saveload(tmp_path, name):
    _require(_DEP[name])
    from xai_ids.models import evaluate_classification

    X, y = _data()
    fast = {"n_estimators": 20} if name == "xgboost" else {"epochs": 3}
    model = MODELS.get(name)(class_weight=True, params=fast)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    pred = model.predict(X)
    assert set(np.unique(pred).tolist()) <= {0, 1}

    m = evaluate_classification(model, X, y)
    assert set(m) == {"accuracy", "precision", "recall", "f1"}
    assert 0.0 <= m["accuracy"] <= 1.0

    path = tmp_path / f"model_{name}.bin"
    model.save(path)
    reloaded = MODELS.get(name).load(path)
    assert np.allclose(reloaded.predict_proba(X), proba, atol=1e-4)
