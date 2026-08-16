"""Explanation stability metrics: does the feature ranking survive (mean over rows)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def spearman_stability(B: np.ndarray, A: np.ndarray) -> float:
    """Mean per-sample Spearman rank correlation between baseline and adversarial."""
    vals = []
    for b, a in zip(B, A, strict=True):
        rb, ra = rankdata(b), rankdata(a)
        if rb.std() == 0 or ra.std() == 0:
            vals.append(1.0 if np.allclose(b, a) else 0.0)
        else:
            vals.append(float(np.corrcoef(rb, ra)[0, 1]))
    return float(np.mean(vals))


def topk_jaccard(B: np.ndarray, A: np.ndarray, k: int) -> float:
    """Mean per-sample Jaccard overlap of the top-k features by |attribution|."""
    vals = []
    for b, a in zip(B, A, strict=True):
        tb = set(np.argsort(-np.abs(b))[:k].tolist())
        ta = set(np.argsort(-np.abs(a))[:k].tolist())
        union = len(tb | ta)
        vals.append(len(tb & ta) / union if union else 1.0)
    return float(np.mean(vals))
