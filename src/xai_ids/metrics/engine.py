"""Metric engine: turn baseline vs adversarial attributions into the metrics row.

Computes drift, stability, and fidelity (on both baseline and adversarial), plus
per-flow-derived tamper outcomes. The caller selects either every attacked flow or only the
prediction-preserved cohort; the runner records both explicitly. Noise-floor
and target-success semantics are upgraded in later improvement phases.
"""

from __future__ import annotations

import numpy as np

from .drift import (
    cosine_drift,
    cosine_drift_per_flow,
    l1_drift,
    l1_drift_per_flow,
    l2_drift,
    l2_drift_per_flow,
)
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
    tamper_outcomes: dict[str, np.ndarray] | None = None,
    fidelity_baseline: np.ndarray | None = None,
    feature_groups: list[list[int]] | None = None,
    benign_drift_samples: dict[str, np.ndarray] | None = None,
    require_prediction_preserved: bool = True,
) -> dict:
    m = metrics_cfg
    preserved = np.asarray(preserved, dtype=bool)
    retention = float(preserved.mean()) if len(preserved) else 0.0

    # The runner invokes this once per cohort. Keeping selection here makes the
    # metric implementation reusable while avoiding any ambiguous strand-specific
    # default in the persisted results.
    keep = preserved if require_prediction_preserved else np.ones(len(preserved), dtype=bool)

    out: dict[str, float] = {
        "n_evaluated": int(keep.sum()),
        "retention_rate": retention,
    }
    if keep.sum() == 0:
        return out

    B, A = baseline[keep], adversarial[keep]
    Xb, Xa = np.asarray(X_base)[keep], np.asarray(X_adv)[keep]

    # --- drift -----------------------------------------------------------------
    drift_fns = {"cosine": cosine_drift, "l1": l1_drift, "l2": l2_drift}
    drift_per_flow_fns = {
        "cosine": cosine_drift_per_flow,
        "l1": l1_drift_per_flow,
        "l2": l2_drift_per_flow,
    }
    for name in m.drift:
        out[f"drift_{name}"] = drift_fns[name](B, A)

    # Benign LIME drift is supplied by the runner as independent fresh-seed
    # realisations on the full attack subset. Slice it to this explicit cohort,
    # then report distributions rather than subtracting/clamping a single value.
    if benign_drift_samples is not None:
        for name in m.drift:
            samples = np.asarray(benign_drift_samples[name], dtype=float)
            if samples.ndim != 2 or samples.shape[1] != len(preserved):
                raise ValueError("benign drift samples must have shape (repeats, attacked_flows)")
            cohort_samples = samples[:, keep]
            raw_per_flow = drift_per_flow_fns[name](B, A)
            effect = raw_per_flow[None, :] - cohort_samples
            out[f"noise_floor_{name}_mean"] = float(cohort_samples.mean())
            out[f"noise_floor_{name}_p025"] = float(np.quantile(cohort_samples, 0.025))
            out[f"noise_floor_{name}_p975"] = float(np.quantile(cohort_samples, 0.975))
            out[f"attack_minus_noise_{name}_mean"] = float(effect.mean())
            out[f"attack_minus_noise_{name}_p025"] = float(np.quantile(effect, 0.025))
            out[f"attack_minus_noise_{name}_p975"] = float(np.quantile(effect, 0.975))
        out["noise_floor_repeats"] = int(next(iter(benign_drift_samples.values())).shape[0])

    # --- stability -------------------------------------------------------------
    if "spearman" in m.stability:
        out["spearman"] = spearman_stability(B, A)
    if "topk_jaccard" in m.stability:
        out["topk_jaccard"] = topk_jaccard(B, A, m.topk)

    # --- fidelity (baseline and adversarial) -----------------------------------
    if "deletion" in m.fidelity:
        out["fidelity_baseline_deletion"] = deletion_auc(
            B, Xb, model, fidelity_baseline, feature_groups
        )
        out["fidelity_adversarial_deletion"] = deletion_auc(
            A, Xa, model, fidelity_baseline, feature_groups
        )
    if "insertion" in m.fidelity:
        out["fidelity_baseline_insertion"] = insertion_auc(
            B, Xb, model, fidelity_baseline, feature_groups
        )
        out["fidelity_adversarial_insertion"] = insertion_auc(
            A, Xa, model, fidelity_baseline, feature_groups
        )

    # --- tamper outcomes ---------------------------------------------------------
    # These are calculated once from the exact baseline realization used to choose
    # the attack target, then sliced into the all/preserved reporting cohorts here.
    if tamper_outcomes is not None:
        selected = {
            name: np.asarray(values)[keep] for name, values in tamper_outcomes.items()
        }
        expected = len(B)
        if any(len(values) != expected for values in selected.values()):
            raise ValueError("tamper outcomes must align with every attacked flow")
        out["tamper_success_topk_displacement"] = float(
            np.mean(selected["success_topk_displacement"])
        )
        for name in (
            "baseline_abs_attribution",
            "adversarial_abs_attribution",
            "relative_attribution_reduction",
            "baseline_rank",
            "adversarial_rank",
            "rank_deterioration",
            "in_topk_baseline",
            "in_topk_adversarial",
        ):
            values = selected[name].astype(float)
            finite = np.isfinite(values)
            out[f"target_{name}"] = float(np.mean(values[finite])) if finite.any() else float("nan")

    return out
