"""Tests for the SPSA estimator and the PGD/FGSM attack."""

from __future__ import annotations

import numpy as np
import pytest

import xai_ids.attacks  # noqa: F401 - registers attacks
import xai_ids.explainers  # noqa: F401 - registers explainers
import xai_ids.models  # noqa: F401 - registers models
from xai_ids.attacks.spsa import spsa_gradient
from xai_ids.oracles import ExplanationOracle, ProbabilityOracle
from xai_ids.registry import ATTACKS, EXPLAINERS, MODELS


def _require(module: str) -> None:
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{module} unavailable: {exc}")


def _fitted_xgb(n=160, d=6, seed=0):
    _require("xgboost")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    model = MODELS.get("xgboost")(class_weight=True, params={"n_estimators": 30})
    model.fit(X, y)
    return model, X, y


def test_attacks_registered():
    assert "pgd" in ATTACKS
    assert "fgsm" in ATTACKS


def test_spsa_direction():
    # f(x) = sum(x): gradient is +1 in every dim; SPSA should point positive.
    rng = np.random.default_rng(0)
    X = np.zeros((50, 4))
    grad, calls = spsa_gradient(lambda z: z.sum(axis=1), X, q=8, delta=0.05, rng=rng)
    assert calls == 16
    assert (grad.mean(axis=0) > 0).all()


def test_pgd_within_epsilon_and_fragility_mask():
    model, X, _ = _fitted_xgb()
    atk = ATTACKS.get("pgd")(
        epsilon=0.05, steps=10, step_size=0.02, spsa_q=1,
        enforce_prediction_preserved=False,
    )
    res = atk.generate(ProbabilityOracle.from_model(model, budget=None), X[:30], objective="loss")
    assert res.X_adv.shape == (30, X.shape[1])
    assert np.max(np.abs(res.X_adv - X[:30])) <= 0.05 + 1e-6  # within epsilon-ball
    assert res.prediction_preserved.shape == (30,)
    assert 0.0 <= res.retention_rate <= 1.0
    assert res.meta["reference_predictions"] == model.predict(X[:30]).tolist()
    assert res.meta["adversarial_predictions"] == model.predict(res.X_adv).tolist()
    assert not any(res.meta["frozen_due_prediction_flip"])
    assert res.meta["n_perturbed"] == int(
        (~np.isclose(res.X_adv - X[:30], 0.0, atol=1e-12)).any(axis=1).sum()
    )


def test_fgsm_single_step():
    model, X, _ = _fitted_xgb()
    res = ATTACKS.get("fgsm")(epsilon=0.05, steps=1, spsa_q=1).generate(
        ProbabilityOracle.from_model(model, budget=None), X[:20], objective="loss"
    )
    assert res.X_adv.shape == (20, X.shape[1])


def test_tamper_preserves_prediction_and_suppresses_target():
    _require("shap")
    model, X, _ = _fitted_xgb()
    Xs = X[:20]
    y_ref = model.predict(Xs)
    explainer = EXPLAINERS.get("shap")(backend="tree")
    baseline = explainer.explain(model, Xs)
    targets = explainer.target_features(baseline, k=1)[:, 0]

    atk = ATTACKS.get("pgd")(
        epsilon=0.1, steps=15, step_size=0.02, spsa_q=1,
        enforce_prediction_preserved=True,
    )
    res = atk.generate(
        ProbabilityOracle.from_model(model, budget=None),
        Xs,
        objective="attribution",
        explanation_oracle=ExplanationOracle.from_components(
            explainer, model, background=Xs, budget=None
        ),
    )

    # tamper enforces prediction preservation in-loop
    assert (model.predict(res.X_adv) == y_ref).all()
    assert res.meta["adversarial_predictions"] == model.predict(res.X_adv).tolist()
    assert len(res.meta["frozen_due_prediction_flip"]) == len(Xs)
    assert res.meta["n_frozen_due_prediction_flip"] == sum(
        res.meta["frozen_due_prediction_flip"]
    )
    assert res.meta["explanation_queries"] > 0
    assert res.meta["target_objective"] == "suppression"
    assert np.asarray(res.meta["objective_baseline_attributions"]).shape == baseline.shape

    # on average the target feature's attribution magnitude should not increase
    adv = explainer.explain(model, res.X_adv)
    idx = np.arange(len(targets))
    reduction = np.abs(baseline[idx, targets]) - np.abs(adv[idx, targets])
    assert reduction.mean() >= -1e-6


def test_rank_margin_tamper_objective_runs_with_query_oracles():
    _require("shap")
    model, X, _ = _fitted_xgb(d=6)
    Xs = X[:8]
    explainer = EXPLAINERS.get("shap")(backend="tree")
    attack = ATTACKS.get("pgd")(
        epsilon=0.05,
        steps=2,
        step_size=0.02,
        spsa_q=1,
        target_objective="rank_margin",
        target_topk=2,
    )
    result = attack.generate(
        ProbabilityOracle.from_model(model, budget=None),
        Xs,
        objective="attribution",
        explanation_oracle=ExplanationOracle.from_components(
            explainer, model, background=Xs, budget=None
        ),
    )

    assert result.meta["target_objective"] == "rank_margin"
    assert result.meta["target_topk"] == 2
    assert len(result.meta["targets"]) == len(Xs)
