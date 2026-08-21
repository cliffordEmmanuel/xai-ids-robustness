"""Fidelity metrics: does an explanation reflect the model's actual decision logic.

Deletion/insertion (Petsiuk et al. 2018; Samek et al. 2017) computed on the
positive-class probability. Both baseline and adversarial explanations use the
same training-derived reference input. One-hot categorical variables are changed
as a group so intermediate inputs remain valid categories.
"""

from __future__ import annotations

import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0 and removed in later 2.x.
# Resolve once, without eagerly touching a name that may not exist.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _auc(probs: np.ndarray) -> float:
    frac = np.linspace(0.0, 1.0, len(probs))
    return float(_trapezoid(probs, frac))


def _pos(model, X: np.ndarray) -> float:
    return float(model.predict_proba(X)[:, 1].mean())


def _baseline(X: np.ndarray, baseline_values) -> np.ndarray:
    """Use a supplied fixed reference; retain a legacy fallback for direct callers."""
    return X.mean(axis=0) if baseline_values is None else np.asarray(baseline_values, dtype=float)


def _groups(n_features: int, feature_groups: list[list[int]] | None) -> list[np.ndarray]:
    """Validate groups, falling back to singleton features for numeric data."""
    groups = feature_groups or [[i] for i in range(n_features)]
    arrays = [np.asarray(group, dtype=int) for group in groups]
    flattened = np.concatenate(arrays) if arrays else np.array([], dtype=int)
    if sorted(flattened.tolist()) != list(range(n_features)):
        raise ValueError("feature_groups must partition every feature exactly once.")
    return arrays


def _group_order(attr: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Rank groups by their largest absolute member attribution per row."""
    scores = np.column_stack([np.abs(attr[:, group]).max(axis=1) for group in groups])
    return np.argsort(-scores, axis=1)


def _replace_group_values(
    matrix: np.ndarray,
    row_groups: np.ndarray,
    groups: list[np.ndarray],
    values: np.ndarray,
) -> None:
    """Set each row's selected group from a fixed reference or source input."""
    for group_index, group in enumerate(groups):
        rows = np.where(row_groups == group_index)[0]
        if len(rows):
            matrix[np.ix_(rows, group)] = values[group]


def deletion_auc(
    attr: np.ndarray,
    X: np.ndarray,
    model,
    baseline_values=None,
    feature_groups: list[list[int]] | None = None,
) -> float:
    """Lower is better: faithful explanations drop the prediction fast when removed."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    base = _baseline(X, baseline_values)
    groups = _groups(d, feature_groups)
    order = _group_order(attr, groups)
    Xm = X.copy()
    probs = [_pos(model, Xm)]
    for m in range(len(groups)):
        _replace_group_values(Xm, order[:, m], groups, base)
        probs.append(_pos(model, Xm))
    return _auc(np.asarray(probs))


def insertion_auc(
    attr: np.ndarray,
    X: np.ndarray,
    model,
    baseline_values=None,
    feature_groups: list[list[int]] | None = None,
) -> float:
    """Higher is better: faithful explanations raise the prediction fast when added."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    base = _baseline(X, baseline_values)
    groups = _groups(d, feature_groups)
    order = _group_order(attr, groups)
    Xm = np.tile(base, (n, 1))  # start from the neutral baseline
    probs = [_pos(model, Xm)]
    for m in range(len(groups)):
        for group_index, group in enumerate(groups):
            rows = np.where(order[:, m] == group_index)[0]
            if len(rows):
                Xm[np.ix_(rows, group)] = X[np.ix_(rows, group)]
        probs.append(_pos(model, Xm))
    return _auc(np.asarray(probs))
