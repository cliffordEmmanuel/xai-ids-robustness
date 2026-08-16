"""Tests for the metric functions and the engine orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from xai_ids.config import MetricsConfig
from xai_ids.metrics import compute_metrics
from xai_ids.metrics.drift import cosine_drift, l1_drift, l2_drift
from xai_ids.metrics.stability import spearman_stability, topk_jaccard


def _require(module: str) -> None:
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{module} unavailable: {exc}")


def test_drift_identical_is_zero():
    B = np.random.default_rng(0).normal(size=(10, 6))
    assert cosine_drift(B, B) == pytest.approx(0.0, abs=1e-9)
    assert l1_drift(B, B) == pytest.approx(0.0, abs=1e-9)
    assert l2_drift(B, B) == pytest.approx(0.0, abs=1e-9)


def test_stability_identical_is_one():
    B = np.random.default_rng(1).normal(size=(10, 6))
    assert spearman_stability(B, B) == pytest.approx(1.0, abs=1e-9)
    assert topk_jaccard(B, B, k=3) == pytest.approx(1.0, abs=1e-9)


def test_topk_jaccard_disjoint():
    B = np.array([[3.0, 2.0, 1.0, 0.0]])
    A = np.array([[0.0, 1.0, 2.0, 3.0]])  # top-1 differs (idx 0 vs idx 3)
    assert topk_jaccard(B, A, k=1) == pytest.approx(0.0)


def _fitted_xgb(n=120, d=6, seed=0):
    _require("xgboost")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    from xai_ids.registry import MODELS

    model = MODELS.get("xgboost")(class_weight=True, params={"n_estimators": 20})
    model.fit(X, y)
    return model, X, y


def test_engine_end_to_end():
    _require("shap")
    import xai_ids.explainers  # noqa: F401
    import xai_ids.models  # noqa: F401
    from xai_ids.registry import EXPLAINERS

    model, X, _ = _fitted_xgb()
    Xs = X[:20]
    expl = EXPLAINERS.get("shap")(backend="tree")
    B = expl.explain(model, Xs)
    A = B + 0.01 * np.random.default_rng(0).normal(size=B.shape)  # small drift
    targets = expl.target_features(B, k=1)[:, 0]
    preserved = np.ones(len(Xs), dtype=bool)

    cfg = MetricsConfig()  # defaults: deletion+insertion, spearman+topk, cosine/l1/l2
    out = compute_metrics(
        B, A, X_base=Xs, X_adv=Xs, model=model,
        preserved=preserved, metrics_cfg=cfg, explainer=expl, targets=targets,
    )
    for key in ("drift_cosine", "drift_l1", "drift_l2", "spearman", "topk_jaccard",
                "fidelity_baseline_deletion", "fidelity_adversarial_insertion",
                "tamper_success", "n_evaluated", "retention_rate"):
        assert key in out
    assert out["n_evaluated"] == 20
    assert 0.0 <= out["tamper_success"] <= 1.0
    # TreeSHAP is deterministic -> noise floor ~ 0
    assert out["noise_floor_cosine"] == pytest.approx(0.0, abs=1e-6)
