"""Attack objectives (both framed so the attack *maximises* them).

* loss (fragility): push the prediction toward the other class.
* attribution (tamper): suppress the target feature's attribution magnitude.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def loss_objective(model, y_ref: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Maximise the probability of the class other than ``y_ref`` (toward a flip)."""
    other = 1 - np.asarray(y_ref)
    idx = np.arange(len(other))

    def f(X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[idx, other]

    return f


def attribution_objective(
    model, explainer, targets: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Maximise suppression of each sample's target feature: f = -|attr[i, t_i]|."""
    idx = np.arange(len(targets))

    def f(X: np.ndarray) -> np.ndarray:
        attr = explainer.explain(model, X)
        return -np.abs(attr[idx, targets])

    return f
