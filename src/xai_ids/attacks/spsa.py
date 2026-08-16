"""SPSA gradient estimator (simultaneous perturbation stochastic approximation).

Estimates the gradient of a scalar objective ``f`` (which the attack *maximises*)
from query-only access — the black-box, model-agnostic core used for both
XGBoost and the MLP. Each direction uses an antithetic Rademacher probe (two
evaluations); ``q`` directions are averaged to reduce variance.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def spsa_gradient(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    q: int,
    delta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Estimate d f / d X per row.

    ``f`` maps ``(n, d) -> (n,)``. Returns ``(gradient (n, d), n_objective_calls)``.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    grad = np.zeros((n, d))
    calls = 0
    for _ in range(q):
        pert = rng.choice([-1.0, 1.0], size=(n, d))  # Rademacher ±1
        f_plus = f(X + delta * pert)
        f_minus = f(X - delta * pert)
        calls += 2
        # (f+ - f-) / (2*delta) * (1/pert); for ±1, 1/pert == pert
        grad += ((f_plus - f_minus) / (2.0 * delta))[:, None] * pert
    return grad / q, calls
