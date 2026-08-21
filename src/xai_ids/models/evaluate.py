"""Classification metrics for the trained IDS models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from ..interfaces.model import Model

MODEL_QUALITY_MIN_ATTACK_RECALL = 0.80
MODEL_QUALITY_MIN_F1 = 0.80


def evaluate_classification(model: Model, X: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Binary test metrics and confusion-matrix counts (positive = attack)."""
    y = np.asarray(y, dtype=int)
    pred = model.predict(X)
    tn = int(np.count_nonzero((y == 0) & (pred == 0)))
    fp = int(np.count_nonzero((y == 0) & (pred == 1)))
    fn = int(np.count_nonzero((y == 1) & (pred == 0)))
    tp = int(np.count_nonzero((y == 1) & (pred == 1)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "test_n_total": int(len(y)),
        "test_n_benign": int(np.count_nonzero(y == 0)),
        "test_n_attack": int(np.count_nonzero(y == 1)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def assess_model_quality(metrics: dict[str, float | int]) -> dict[str, float | int | bool]:
    """Apply the preregistered interpretation threshold without dropping evidence.

    A run is eligible for explanatory comparisons only when the IDS detects the
    attack class with both recall and F1 of at least 0.80. The runner still
    persists every run below this threshold so model-quality failure remains
    visible rather than being silently filtered away.
    """
    return {
        **metrics,
        "min_attack_recall": MODEL_QUALITY_MIN_ATTACK_RECALL,
        "min_f1": MODEL_QUALITY_MIN_F1,
        "comparison_eligible": bool(
            float(metrics["recall"]) >= MODEL_QUALITY_MIN_ATTACK_RECALL
            and float(metrics["f1"]) >= MODEL_QUALITY_MIN_F1
        ),
    }
