"""Attribution drift metrics, with per-flow values for uncertainty analysis."""

from __future__ import annotations

import numpy as np


def cosine_drift_per_flow(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Cosine attribution drift for each flow (zero vectors count as identical)."""
    bn = np.linalg.norm(B, axis=1)
    an = np.linalg.norm(A, axis=1)
    denom = bn * an
    dot = (B * A).sum(axis=1)
    # zero vectors -> treat as identical direction (distance 0)
    cos = np.where(denom > 0, dot / np.where(denom > 0, denom, 1.0), 1.0)
    return 1.0 - cos


def cosine_drift(B: np.ndarray, A: np.ndarray) -> float:
    return float(np.mean(cosine_drift_per_flow(B, A)))


def l1_drift_per_flow(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """L1 attribution drift for each flow."""
    return np.abs(B - A).sum(axis=1)


def l1_drift(B: np.ndarray, A: np.ndarray) -> float:
    return float(np.mean(l1_drift_per_flow(B, A)))


def l2_drift_per_flow(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """L2 attribution drift for each flow."""
    return np.linalg.norm(B - A, axis=1)


def l2_drift(B: np.ndarray, A: np.ndarray) -> float:
    return float(np.mean(l2_drift_per_flow(B, A)))
