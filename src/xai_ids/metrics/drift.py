"""Attribution drift metrics: how far the attribution vector moved (mean over rows)."""

from __future__ import annotations

import numpy as np


def cosine_drift(B: np.ndarray, A: np.ndarray) -> float:
    bn = np.linalg.norm(B, axis=1)
    an = np.linalg.norm(A, axis=1)
    denom = bn * an
    dot = (B * A).sum(axis=1)
    # zero vectors -> treat as identical direction (distance 0)
    cos = np.where(denom > 0, dot / np.where(denom > 0, denom, 1.0), 1.0)
    return float(np.mean(1.0 - cos))


def l1_drift(B: np.ndarray, A: np.ndarray) -> float:
    return float(np.mean(np.abs(B - A).sum(axis=1)))


def l2_drift(B: np.ndarray, A: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(B - A, axis=1)))
