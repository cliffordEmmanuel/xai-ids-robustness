"""Tests for the SHAP and LIME explainers on a small XGBoost model."""

from __future__ import annotations

import numpy as np
import pytest

import xai_ids.explainers  # noqa: F401 - registers the explainers
import xai_ids.models  # noqa: F401 - registers the models
from xai_ids.registry import EXPLAINERS, MODELS


def _require(module: str) -> None:
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{module} unavailable: {exc}")


def _fitted_xgb(n=120, d=6, seed=0):
    _require("xgboost")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    model = MODELS.get("xgboost")(class_weight=True, params={"n_estimators": 20})
    model.fit(X, y)
    return model, X, y


def test_explainers_registered():
    assert "shap" in EXPLAINERS
    assert "lime" in EXPLAINERS


def test_shap_tree_shape():
    _require("shap")
    model, X, _ = _fitted_xgb()
    attrs = EXPLAINERS.get("shap")(backend="tree").explain(model, X[:20], background=X)
    assert attrs.shape == (20, X.shape[1])
    assert np.isfinite(attrs).all()


def test_shap_kernel_shape():
    _require("shap")
    model, X, _ = _fitted_xgb()
    expl = EXPLAINERS.get("shap")(
        backend="kernel", background_size=20, kernel_nsamples=50, kernel_l1_reg=0.0001
    )
    attrs = expl.explain(model, X[:5], background=X)
    assert attrs.shape == (5, X.shape[1])
    assert np.isfinite(attrs).all()
    diagnostics = expl.numerical_diagnostics()
    assert diagnostics["attributions_finite"] is True
    assert diagnostics["max_additivity_error"] < 1e-6


def test_lime_shape():
    _require("lime")
    model, X, _ = _fitted_xgb()
    expl = EXPLAINERS.get("lime")(num_samples=100, background_size=100)
    attrs = expl.explain(model, X[:5], background=X)
    assert attrs.shape == (5, X.shape[1])
    assert np.isfinite(attrs).all()


def test_target_features():
    _require("shap")
    model, X, _ = _fitted_xgb()
    attrs = EXPLAINERS.get("shap")(backend="tree").explain(model, X[:10], background=X)
    top1 = EXPLAINERS.get("shap")(backend="tree").target_features(attrs, k=1)
    assert top1.shape == (10, 1)
    assert top1.max() < X.shape[1]
