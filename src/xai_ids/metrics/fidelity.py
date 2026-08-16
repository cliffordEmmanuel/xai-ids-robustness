"""Fidelity metrics: does an explanation reflect the model's actual decision logic.

Deletion/insertion (Petsiuk et al. 2018; Samek et al. 2017) computed on the
positive-class probability. Features are removed/added in order of |attribution|;
the "removed" value is 0 (the mean in standardised feature space). Batched: one
predict_proba per step.
"""

from __future__ import annotations

import numpy as np


def _auc(probs: np.ndarray) -> float:
    frac = np.linspace(0.0, 1.0, len(probs))
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(probs, frac))


def _pos(model, X: np.ndarray) -> float:
    return float(model.predict_proba(X)[:, 1].mean())


def deletion_auc(attr: np.ndarray, X: np.ndarray, model) -> float:
    """Lower is better: faithful explanations drop the prediction fast when removed."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    order = np.argsort(-np.abs(attr), axis=1)
    rows = np.arange(n)
    Xm = X.copy()
    probs = [_pos(model, Xm)]
    for m in range(d):
        Xm[rows, order[:, m]] = 0.0
        probs.append(_pos(model, Xm))
    return _auc(np.asarray(probs))


def insertion_auc(attr: np.ndarray, X: np.ndarray, model) -> float:
    """Higher is better: faithful explanations raise the prediction fast when added."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    order = np.argsort(-np.abs(attr), axis=1)
    rows = np.arange(n)
    Xm = np.zeros_like(X)
    probs = [_pos(model, Xm)]
    for m in range(d):
        Xm[rows, order[:, m]] = X[rows, order[:, m]]
        probs.append(_pos(model, Xm))
    return _auc(np.asarray(probs))
