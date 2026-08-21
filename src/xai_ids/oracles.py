"""Restricted query interfaces used by black-box attack implementations.

The benchmark runner owns the trained model, explainer, and training background.
Attacks receive only these two narrow services:

* :class:`ProbabilityOracle` exposes class probabilities (and labels derived from
  those probabilities), counting one model query per input row;
* :class:`ExplanationOracle` exposes attributions for a supplied input batch,
  counting one logical explanation-service request per call.

The latter deliberately does *not* estimate the hidden model calls made inside
TreeSHAP, KernelSHAP, or LIME. Those backend-specific calls are not observable at
the service boundary and are recorded as excluded from the public query count.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class QueryBudgetExceeded(RuntimeError):
    """Raised before an oracle call that would exceed its configured budget."""


class ProbabilityOracle:
    """Probability-only model access for an attack.

    The wrapped callable is intentionally private; the public API exposes no
    trained-model object, weights, native backend, or direct hard-label method.
    Labels are calculated from the allowed probability response.
    """

    __slots__ = ("_predict_proba", "_budget", "_queries")

    def __init__(self, predict_proba: Callable[[np.ndarray], np.ndarray], *, budget: int | None):
        self._predict_proba = predict_proba
        self._budget = budget
        self._queries = 0

    @classmethod
    def from_model(cls, model, *, budget: int | None) -> ProbabilityOracle:
        """Bind only the model's probability callable into the attack service."""
        return cls(model.predict_proba, budget=budget)

    @property
    def queries(self) -> int:
        """Number of individual input probability vectors requested so far."""
        return self._queries

    @property
    def budget(self) -> int | None:
        return self._budget

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities, charging one black-box query per input row."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("ProbabilityOracle expects a two-dimensional input batch.")
        self._consume(len(X))
        probabilities = np.asarray(self._predict_proba(X), dtype=float)
        if probabilities.ndim != 2 or len(probabilities) != len(X):
            raise ValueError("Probability oracle returned an invalid probability array.")
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Derive labels from the permitted probability response."""
        return np.argmax(self.predict_proba(X), axis=1)

    def record(self) -> dict:
        """Return serialisable accounting information for an attack artifact."""
        return {
            "access": "probability_query",
            "queries": self.queries,
            "budget": self.budget,
            "query_unit": "individual input probability vector",
        }

    def _consume(self, amount: int) -> None:
        if self._budget is not None and self._queries + amount > self._budget:
            raise QueryBudgetExceeded(
                "model-query budget exceeded: "
                f"requested {amount}, used {self._queries}, budget {self._budget}"
            )
        self._queries += amount


class ExplanationOracle:
    """Explanation-service access for an attack, without model/background access."""

    __slots__ = ("_explain", "_budget", "_queries")

    def __init__(self, explain: Callable[[np.ndarray], np.ndarray], *, budget: int | None):
        self._explain = explain
        self._budget = budget
        self._queries = 0

    @classmethod
    def from_components(
        cls, explainer, model, *, background: np.ndarray | None, budget: int | None
    ) -> ExplanationOracle:
        """Create a service closure owned by the runner, not the attack.

        TreeSHAP may use privileged model structure inside this service. That is
        an implementation detail of the explanation provider; the attacker sees
        only the returned attribution array.
        """

        def explain(X: np.ndarray) -> np.ndarray:
            return explainer.explain(model, X, background=background)

        return cls(explain, budget=budget)

    @property
    def queries(self) -> int:
        """Number of logical requests made to the explanation service."""
        return self._queries

    @property
    def budget(self) -> int | None:
        return self._budget

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Return attributions for a batch, charging one service request."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("ExplanationOracle expects a two-dimensional input batch.")
        self._consume()
        attributions = np.asarray(self._explain(X), dtype=float)
        if attributions.ndim != 2 or attributions.shape != X.shape:
            raise ValueError("Explanation oracle returned an invalid attribution array.")
        return attributions

    def record(self) -> dict:
        """Return serialisable accounting and its explicit measurement boundary."""
        return {
            "access": "explanation_oracle",
            "queries": self.queries,
            "budget": self.budget,
            "query_unit": "logical explanation-service request per batch",
            "internal_model_queries": "not_observed_at_oracle_boundary",
        }

    def _consume(self) -> None:
        if self._budget is not None and self._queries + 1 > self._budget:
            raise QueryBudgetExceeded(
                "explanation-query budget exceeded: "
                f"used {self._queries}, budget {self._budget}"
            )
        self._queries += 1
