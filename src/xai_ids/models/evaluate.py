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


def evaluate_classification(model: Model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Accuracy / precision / recall / F1 on ``(X, y)`` (binary, positive = attack)."""
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
