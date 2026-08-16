"""Metric engine: turn baseline vs adversarial attributions into the metrics row.

Computes drift, stability, and fidelity (on both baseline and adversarial), plus
tamper success, with the safeguards: cohort filtering (prediction-preserved rows),
noise-floor subtraction for stochastic explainers, and objective/evaluation
separation (tamper_success kept as a distinct field, not a robustness metric).
"""

from __future__ import annotations

import numpy as np

from .drift import cosine_drift, l1_drift, l2_drift
from .fidelity import deletion_auc, insertion_auc
from .stability import spearman_stability, topk_jaccard


def compute_metrics(
    baseline: np.ndarray,
    adversarial: np.ndarray,
    *,
    X_base: np.ndarray,
    X_adv: np.ndarray,
    model,
    preserved: np.ndarray,
    metrics_cfg,
    explainer=None,
    targets=None,
) -> dict:
    m = metrics_cfg
    keep = np.asarray(preserved, dtype=bool)

    out: dict[str, float] = {
        "n_evaluated": int(keep.sum()),
        "retention_rate": float(keep.mean()) if len(keep) else 0.0,
    }
    if keep.sum() == 0:
        return out

    B, A = baseline[keep], adversarial[keep]
    Xb, Xa = np.asarray(X_base)[keep], np.asarray(X_adv)[keep]

    # --- drift -----------------------------------------------------------------
    drift_fns = {"cosine": cosine_drift, "l1": l1_drift, "l2": l2_drift}
    for name in m.drift:
        out[f"drift_{name}"] = drift_fns[name](B, A)

    # noise-floor subtraction for stochastic explainers (re-explain the baseline)
    if m.subtract_lime_noise_floor and explainer is not None:
        B2 = explainer.explain(model, Xb)
        for name in m.drift:
            nf = drift_fns[name](B, B2)
            out[f"noise_floor_{name}"] = nf
            out[f"drift_{name}_adj"] = out[f"drift_{name}"] - nf

    # --- stability -------------------------------------------------------------
    if "spearman" in m.stability:
        out["spearman"] = spearman_stability(B, A)
    if "topk_jaccard" in m.stability:
        out["topk_jaccard"] = topk_jaccard(B, A, m.topk)

    # --- fidelity (baseline and adversarial) -----------------------------------
    if "deletion" in m.fidelity:
        out["fidelity_baseline_deletion"] = deletion_auc(B, Xb, model)
        out["fidelity_adversarial_deletion"] = deletion_auc(A, Xa, model)
    if "insertion" in m.fidelity:
        out["fidelity_baseline_insertion"] = insertion_auc(B, Xb, model)
        out["fidelity_adversarial_insertion"] = insertion_auc(A, Xa, model)

    # --- tamper success (kept separate from robustness metrics) ----------------
    if targets is not None:
        t = np.asarray(targets)[keep]
        success = [
            int(t[i] not in set(np.argsort(-np.abs(A[i]))[: m.topk].tolist()))
            for i in range(len(t))
        ]
        out["tamper_success"] = float(np.mean(success))

    return out
