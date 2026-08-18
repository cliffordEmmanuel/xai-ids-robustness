"""Tests for the metric functions and the engine orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from xai_ids.config import MetricsConfig
from xai_ids.metrics import compute_metrics
from xai_ids.metrics.drift import cosine_drift, l1_drift, l2_drift
from xai_ids.metrics.fidelity import deletion_auc, insertion_auc
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


class _RecordingModel:
    def __init__(self):
        self.calls: list[np.ndarray] = []

    def predict_proba(self, X):
        self.calls.append(np.asarray(X).copy())
        p = np.clip(np.asarray(X).mean(axis=1), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def test_fidelity_changes_one_hot_features_as_a_valid_group():
    X = np.array([[1.0, 0.0, 0.4], [0.0, 1.0, 0.6]])
    attr = np.array([[5.0, 0.1, 1.0], [0.1, 5.0, 1.0]])
    reference = np.array([1.0, 0.0, 0.5])
    groups = [[0, 1], [2]]
    model = _RecordingModel()

    deletion_auc(attr, X, model, reference, groups)
    insertion_auc(attr, X, model, reference, groups)

    for queried in model.calls:
        categorical = queried[:, :2]
        assert np.allclose(categorical.sum(axis=1), 1.0)
        assert np.isin(categorical, [0.0, 1.0]).all()


def test_engine_uses_same_supplied_reference_for_both_fidelity_paths(monkeypatch):
    from xai_ids import metrics

    seen = []

    def capture(attr, X, model, baseline_values=None, feature_groups=None):
        seen.append((np.asarray(baseline_values).copy(), feature_groups))
        return 0.5

    monkeypatch.setattr(metrics.engine, "deletion_auc", capture)
    monkeypatch.setattr(metrics.engine, "insertion_auc", capture)
    X = np.array([[0.2, 0.8], [0.4, 0.6]])
    attrs = np.array([[1.0, 0.0], [0.0, 1.0]])
    reference = np.array([0.3, 0.7])
    groups = [[0], [1]]
    out = metrics.compute_metrics(
        attrs,
        attrs,
        X_base=X,
        X_adv=X + 0.1,
        model=_RecordingModel(),
        preserved=np.ones(len(X), dtype=bool),
        metrics_cfg=MetricsConfig(),
        fidelity_baseline=reference,
        feature_groups=groups,
    )

    assert len(seen) == 4
    assert all(np.array_equal(baseline, reference) for baseline, _ in seen)
    assert all(feature_groups == groups for _, feature_groups in seen)
    assert out["fidelity_baseline_deletion"] == 0.5


def test_engine_reports_benign_noise_distribution_without_clamping_effect():
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    samples = {"l1": np.array([[0.2, 0.4], [0.6, 0.8]])}
    out = compute_metrics(
        B,
        A,
        X_base=B,
        X_adv=A,
        model=_RecordingModel(),
        preserved=np.array([True, False]),
        metrics_cfg=MetricsConfig(drift=["l1"], stability=[], fidelity=[]),
        benign_drift_samples=samples,
        require_prediction_preserved=True,
    )

    assert out["drift_l1"] == pytest.approx(2.0)
    assert out["noise_floor_l1_mean"] == pytest.approx(0.4)
    assert out["attack_minus_noise_l1_mean"] == pytest.approx(1.6)
    assert out["noise_floor_repeats"] == 2
    assert "drift_l1_adj" not in out


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
    preserved = np.ones(len(Xs), dtype=bool)
    tamper_outcomes = {
        "success_topk_displacement": np.zeros(len(Xs), dtype=bool),
        "baseline_abs_attribution": np.ones(len(Xs)),
        "adversarial_abs_attribution": np.full(len(Xs), 0.5),
        "relative_attribution_reduction": np.full(len(Xs), 0.5),
        "baseline_rank": np.ones(len(Xs)),
        "adversarial_rank": np.full(len(Xs), 2),
        "rank_deterioration": np.ones(len(Xs)),
        "in_topk_baseline": np.ones(len(Xs), dtype=bool),
        "in_topk_adversarial": np.ones(len(Xs), dtype=bool),
    }

    cfg = MetricsConfig()  # defaults: deletion+insertion, spearman+topk, cosine/l1/l2
    out = compute_metrics(
        B, A, X_base=Xs, X_adv=Xs, model=model,
        preserved=preserved, metrics_cfg=cfg, tamper_outcomes=tamper_outcomes,
    )
    for key in ("drift_cosine", "drift_l1", "drift_l2", "spearman", "topk_jaccard",
                "fidelity_baseline_deletion", "fidelity_adversarial_insertion",
                "tamper_success_topk_displacement", "target_rank_deterioration",
                "n_evaluated", "retention_rate"):
        assert key in out
    assert out["n_evaluated"] == 20
    assert 0.0 <= out["tamper_success_topk_displacement"] <= 1.0
