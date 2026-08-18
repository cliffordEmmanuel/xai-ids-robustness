"""Tests for the explicit black-box attack service boundaries."""

from __future__ import annotations

import numpy as np
import pytest

import xai_ids.attacks  # noqa: F401 - registers PGD
from xai_ids.oracles import ExplanationOracle, ProbabilityOracle, QueryBudgetExceeded
from xai_ids.registry import ATTACKS


class _NativeForbiddenModel:
    """A stand-in proving oracle-attacked code never needs model internals."""

    @property
    def native(self):  # pragma: no cover - executing it is the failure condition
        raise AssertionError("native model state must not be accessed")

    def predict(self, X):  # pragma: no cover - labels must derive from probabilities
        raise AssertionError("raw predict must not be accessed")

    def predict_proba(self, X):
        p = np.clip(np.asarray(X, dtype=float)[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def test_probability_oracle_exposes_only_probability_based_access_and_counts_rows():
    oracle = ProbabilityOracle.from_model(_NativeForbiddenModel(), budget=3)
    X = np.array([[0.2], [0.8]])

    assert oracle.predict(X).tolist() == [0, 1]
    assert oracle.queries == 2
    assert not hasattr(oracle, "native")
    assert oracle.record()["query_unit"] == "individual input probability vector"

    with pytest.raises(QueryBudgetExceeded, match="model-query budget"):
        oracle.predict_proba(X)
    assert oracle.queries == 2  # rejected calls are not charged


def test_explanation_oracle_enforces_request_budget_and_records_boundary():
    oracle = ExplanationOracle(lambda X: np.asarray(X, dtype=float), budget=1)
    X = np.array([[0.2, 0.4]])

    assert np.array_equal(oracle.explain(X), X)
    assert oracle.queries == 1
    assert oracle.record()["internal_model_queries"] == "not_observed_at_oracle_boundary"
    with pytest.raises(QueryBudgetExceeded, match="explanation-query budget"):
        oracle.explain(X)


def test_attack_rejects_raw_models_but_runs_without_native_model_access():
    model = _NativeForbiddenModel()
    attack = ATTACKS.get("pgd")(
        epsilon=0.01, steps=1, step_size=0.01, spsa_q=1, enforce_prediction_preserved=False
    )
    X = np.array([[0.2], [0.8]])

    with pytest.raises(TypeError, match="ProbabilityOracle"):
        attack.generate(model, X, objective="loss")

    result = attack.generate(ProbabilityOracle.from_model(model, budget=20), X, objective="loss")
    assert result.meta["model_queries"] == 8
    assert result.meta["explanation_queries"] == 0
