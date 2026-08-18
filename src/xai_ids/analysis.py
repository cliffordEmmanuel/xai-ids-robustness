"""Conditional within-cell analysis for completed schema-3.5+ experiments.

The routines here intentionally quantify precision *within* one experimental
cell. They do not turn flows or seeds into independent dataset/model
conditions, and therefore do not support pooled population claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics.drift import cosine_drift_per_flow, l1_drift_per_flow, l2_drift_per_flow
from .results import sha256_file, validate_results_dir

_ANALYSIS_SCHEMA_VERSIONS = frozenset({"3.5", "3.6"})

_DRIFT = {
    "attack_drift_cosine": cosine_drift_per_flow,
    "attack_drift_l1": l1_drift_per_flow,
    "attack_drift_l2": l2_drift_per_flow,
}
_SUMMARY_COLUMNS = (
    "experiment_id",
    "dataset",
    "model",
    "explainer",
    "explainer_backend",
    "strand",
    "method",
    "epsilon",
    "seed",
    "cohort",
    "metric",
    "estimate",
    "ci_low",
    "ci_high",
    "n_flows",
    "bootstrap_repeats",
)
_LIME_TEST_COLUMNS = (
    "experiment_id",
    "dataset",
    "model",
    "strand",
    "method",
    "epsilon",
    "seed",
    "cohort",
    "metric",
    "attack_mean",
    "benign_mean",
    "attack_minus_benign_mean",
    "ci_low",
    "ci_high",
    "p_greater_sign_flip",
    "n_flows",
    "noise_floor_repeats",
    "bootstrap_repeats",
    "permutation_repeats",
)
_KERNEL_STABILITY_COLUMNS = (
    "experiment_id",
    "dataset",
    "model",
    "strand",
    "method",
    "epsilon",
    "seed",
    "shap_kernel_nsamples",
    "shap_kernel_l1_reg",
    "shap_kernel_stability_repeats",
    "shap_kernel_additivity_tolerance",
    "shap_kernel_repeat_cosine_drift_tolerance",
    "require_shap_kernel_stability",
    "n_explainer_convergence_warnings",
    "shap_kernel_attributions_finite",
    "shap_kernel_max_additivity_error",
    "shap_kernel_repeat_cosine_drift_mean",
    "shap_kernel_repeat_cosine_drift_max",
    "shap_kernel_stability_status",
)
_KERNEL_CANDIDATE_COLUMNS = (
    "shap_kernel_nsamples",
    "shap_kernel_l1_reg",
    "n_cells",
    "n_passing_cells",
    "all_cells_pass",
    "all_attributions_finite",
    "max_convergence_warnings",
    "worst_additivity_error",
    "additivity_tolerance",
    "worst_repeat_cosine_drift",
    "repeat_cosine_drift_tolerance",
)


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Paths and row counts written by :func:`analyze_results_dir`."""

    output_dir: Path
    bootstrap_path: Path
    lime_test_path: Path
    kernel_stability_path: Path
    kernel_candidate_path: Path
    manifest_path: Path
    bootstrap_rows: int
    lime_test_rows: int
    kernel_stability_rows: int
    kernel_candidate_rows: int


def _finite(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def bootstrap_mean_ci(
    values: np.ndarray | pd.Series,
    *,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    """Return a non-parametric percentile interval for a conditional cell mean."""
    if repeats < 1:
        raise ValueError("bootstrap repeats must be at least one")
    observed = _finite(values)
    if not len(observed):
        return float("nan"), float("nan"), float("nan"), 0
    indices = rng.integers(0, len(observed), size=(repeats, len(observed)))
    samples = observed[indices].mean(axis=1)
    return (
        float(observed.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        int(len(observed)),
    )


def paired_sign_flip_test(
    differences: np.ndarray | pd.Series,
    *,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Return a one-sided paired sign-flip p-value for attack drift > benign drift."""
    if repeats < 1:
        raise ValueError("permutation repeats must be at least one")
    observed = _finite(differences)
    if not len(observed):
        return float("nan"), float("nan"), 0
    effect = float(observed.mean())
    signs = rng.integers(0, 2, size=(repeats, len(observed)), dtype=np.int8) * 2 - 1
    null_means = (signs * observed).mean(axis=1)
    p_greater = (np.count_nonzero(null_means >= effect) + 1) / (repeats + 1)
    return effect, float(p_greater), int(len(observed))


def _boolean(values: pd.Series, *, name: str) -> np.ndarray:
    normalised = values.astype(str).str.lower()
    mapping = {"true": True, "false": False}
    if not normalised.isin(mapping).all():
        raise ValueError(f"invalid boolean values for {name}")
    return normalised.map(mapping).to_numpy(dtype=bool)


def _attack_drift(path: Path, *, expected_rows: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        required = {"baseline_attributions", "adversarial_attributions"}
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"{path} is missing attribution arrays: {', '.join(missing)}")
        baseline = np.asarray(artifact["baseline_attributions"], dtype=float)
        adversarial = np.asarray(artifact["adversarial_attributions"], dtype=float)
    if baseline.shape != adversarial.shape or baseline.ndim != 2 or len(baseline) != expected_rows:
        raise ValueError(f"{path} attribution arrays do not align with per_flow.csv")
    return {name: fn(baseline, adversarial) for name, fn in _DRIFT.items()}


def _identity(row: pd.Series) -> dict[str, Any]:
    return {
        "experiment_id": row["experiment_id"],
        "dataset": row["dataset"],
        "model": row["model"],
        "explainer": row["explainer"],
        "explainer_backend": row["explainer_backend"],
        "strand": row["strand"],
        "method": row["method"],
        "epsilon": row["epsilon"],
        "seed": row["seed"],
    }


def summarize_experiment(
    row: pd.Series,
    run_dir: str | Path,
    *,
    bootstrap_repeats: int,
    permutation_repeats: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create conditional bootstrap and LIME-noise summaries for one run."""
    run_dir = Path(run_dir)
    flows = pd.read_csv(run_dir / "per_flow.csv")
    preserved = _boolean(flows["prediction_preserved"], name="prediction_preserved")
    attack_drift = _attack_drift(run_dir / "attack.npz", expected_rows=len(flows))
    flow_metrics: dict[str, np.ndarray] = {
        "prediction_preserved": preserved.astype(float),
        "perturbation_linf": flows["perturbation_linf"].to_numpy(dtype=float),
        "perturbation_l1": flows["perturbation_l1"].to_numpy(dtype=float),
        "perturbation_l2": flows["perturbation_l2"].to_numpy(dtype=float),
        **attack_drift,
    }
    if row["strand"] == "tamper":
        for name in (
            "target_success_topk_displacement",
            "target_baseline_abs_attribution",
            "target_adversarial_abs_attribution",
            "target_relative_attribution_reduction",
            "target_baseline_rank",
            "target_adversarial_rank",
            "target_rank_deterioration",
            "target_in_topk_baseline",
            "target_in_topk_adversarial",
        ):
            values = flows[name]
            is_boolean = name in {
                "target_success_topk_displacement",
                "target_in_topk_baseline",
                "target_in_topk_adversarial",
            }
            flow_metrics[name] = (
                _boolean(values, name=name).astype(float)
                if is_boolean
                else values.to_numpy(dtype=float)
            )

    identity = _identity(row)
    bootstrap_rows: list[dict[str, Any]] = []
    for cohort, mask in (("all", np.ones(len(flows), dtype=bool)), ("preserved", preserved)):
        for metric, values in flow_metrics.items():
            estimate, ci_low, ci_high, n_flows = bootstrap_mean_ci(
                values[mask], repeats=bootstrap_repeats, rng=rng
            )
            bootstrap_rows.append(
                {
                    **identity,
                    "cohort": cohort,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_flows": n_flows,
                    "bootstrap_repeats": bootstrap_repeats,
                }
            )

    lime_rows: list[dict[str, Any]] = []
    if row["explainer"] != "lime":
        return bootstrap_rows, lime_rows
    with np.load(run_dir / "noise_floor.npz", allow_pickle=False) as artifact:
        mode = str(artifact["mode"].item())
        if mode != "repeated_benign_lime":
            return bootstrap_rows, lime_rows
        for attack_metric, attack_values in attack_drift.items():
            drift_name = attack_metric.removeprefix("attack_drift_")
            noise = np.asarray(artifact[f"benign_drift_{drift_name}"], dtype=float)
            if noise.ndim != 2 or noise.shape[1] != len(flows):
                raise ValueError(f"noise-floor {drift_name} values do not align with per_flow.csv")
            benign_per_flow = noise.mean(axis=0)
            cohorts = (("all", np.ones(len(flows), dtype=bool)), ("preserved", preserved))
            for cohort, mask in cohorts:
                difference = attack_values[mask] - benign_per_flow[mask]
                effect, p_greater, n_flows = paired_sign_flip_test(
                    difference, repeats=permutation_repeats, rng=rng
                )
                _, ci_low, ci_high, _ = bootstrap_mean_ci(
                    difference, repeats=bootstrap_repeats, rng=rng
                )
                lime_rows.append(
                    {
                        **identity,
                        "cohort": cohort,
                        "metric": drift_name,
                        "attack_mean": (
                            float(np.mean(attack_values[mask])) if mask.any() else float("nan")
                        ),
                        "benign_mean": (
                            float(np.mean(benign_per_flow[mask])) if mask.any() else float("nan")
                        ),
                        "attack_minus_benign_mean": effect,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_greater_sign_flip": p_greater,
                        "n_flows": n_flows,
                        "noise_floor_repeats": int(noise.shape[0]),
                        "bootstrap_repeats": bootstrap_repeats,
                        "permutation_repeats": permutation_repeats,
                    }
                )
    return bootstrap_rows, lime_rows


def _kernel_stability_reports(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-cell and per-candidate KernelSHAP numerical-stability reports.

    This deliberately reads only numerical diagnostics.  Attack outcomes are
    excluded so a configuration cannot be selected because it produces a more
    favourable robustness result.
    """
    if "shap_kernel_stability_status" not in frame.columns:
        return (
            pd.DataFrame(columns=_KERNEL_STABILITY_COLUMNS),
            pd.DataFrame(columns=_KERNEL_CANDIDATE_COLUMNS),
        )
    kernel = frame.loc[
        (frame["explainer"] == "shap") & (frame["explainer_backend"] == "kernel"),
        list(_KERNEL_STABILITY_COLUMNS),
    ].copy()
    if kernel.empty:
        return kernel, pd.DataFrame(columns=_KERNEL_CANDIDATE_COLUMNS)

    for name in (
        "shap_kernel_nsamples",
        "shap_kernel_stability_repeats",
        "n_explainer_convergence_warnings",
        "shap_kernel_additivity_tolerance",
        "shap_kernel_repeat_cosine_drift_tolerance",
        "shap_kernel_max_additivity_error",
        "shap_kernel_repeat_cosine_drift_mean",
        "shap_kernel_repeat_cosine_drift_max",
    ):
        kernel[name] = pd.to_numeric(kernel[name], errors="raise")
    kernel["shap_kernel_attributions_finite"] = _boolean(
        kernel["shap_kernel_attributions_finite"], name="shap_kernel_attributions_finite"
    )
    kernel["require_shap_kernel_stability"] = _boolean(
        kernel["require_shap_kernel_stability"], name="require_shap_kernel_stability"
    )
    kernel["cell_passes_numerical_rule"] = (
        kernel["shap_kernel_stability_status"].eq("pass")
        & kernel["shap_kernel_attributions_finite"]
        & kernel["n_explainer_convergence_warnings"].eq(0)
        & kernel["shap_kernel_max_additivity_error"].le(
            kernel["shap_kernel_additivity_tolerance"]
        )
        & kernel["shap_kernel_repeat_cosine_drift_max"].le(
            kernel["shap_kernel_repeat_cosine_drift_tolerance"]
        )
    )
    candidate_rows: list[dict[str, Any]] = []
    for (nsamples, l1_reg), group in kernel.groupby(
        ["shap_kernel_nsamples", "shap_kernel_l1_reg"], sort=True
    ):
        candidate_rows.append(
            {
                "shap_kernel_nsamples": int(nsamples),
                "shap_kernel_l1_reg": l1_reg,
                "n_cells": int(len(group)),
                "n_passing_cells": int(group["cell_passes_numerical_rule"].sum()),
                "all_cells_pass": bool(group["cell_passes_numerical_rule"].all()),
                "all_attributions_finite": bool(group["shap_kernel_attributions_finite"].all()),
                "max_convergence_warnings": int(group["n_explainer_convergence_warnings"].max()),
                "worst_additivity_error": float(group["shap_kernel_max_additivity_error"].max()),
                "additivity_tolerance": float(group["shap_kernel_additivity_tolerance"].max()),
                "worst_repeat_cosine_drift": float(
                    group["shap_kernel_repeat_cosine_drift_max"].max()
                ),
                "repeat_cosine_drift_tolerance": float(
                    group["shap_kernel_repeat_cosine_drift_tolerance"].max()
                ),
            }
        )
    return kernel, pd.DataFrame(candidate_rows, columns=_KERNEL_CANDIDATE_COLUMNS)


def analyze_results_dir(
    results_dir: str | Path,
    *,
    analysis_dir: str | Path | None = None,
    bootstrap_repeats: int = 10_000,
    permutation_repeats: int = 10_000,
    seed: int = 2026,
) -> AnalysisArtifacts:
    """Validate and analyse a completed schema-3.5 or 3.6 result directory."""
    results_dir = Path(results_dir)
    report = validate_results_dir(results_dir)
    if not report.ok:
        raise ValueError("results validation failed:\n- " + "\n- ".join(report.errors))
    frame = pd.read_csv(results_dir / "results.csv")
    schemas = set(frame["schema_version"].astype(str))
    if len(schemas) != 1 or not schemas <= _ANALYSIS_SCHEMA_VERSIONS:
        raise ValueError(
            "analysis requires one supported schema "
            f"{sorted(_ANALYSIS_SCHEMA_VERSIONS)}; found {sorted(schemas)}"
        )
    schema_version = next(iter(schemas))

    output_dir = Path(analysis_dir) if analysis_dir is not None else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, Any]] = []
    lime_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        summaries, lime_tests = summarize_experiment(
            row,
            results_dir / str(row["experiment_id"]),
            bootstrap_repeats=bootstrap_repeats,
            permutation_repeats=permutation_repeats,
            rng=rng,
        )
        bootstrap_rows.extend(summaries)
        lime_rows.extend(lime_tests)

    bootstrap_path = output_dir / "per_flow_bootstrap.csv"
    lime_test_path = output_dir / "lime_attack_vs_noise.csv"
    pd.DataFrame(bootstrap_rows, columns=_SUMMARY_COLUMNS).to_csv(bootstrap_path, index=False)
    pd.DataFrame(lime_rows, columns=_LIME_TEST_COLUMNS).to_csv(lime_test_path, index=False)
    kernel_stability, kernel_candidates = _kernel_stability_reports(frame)
    kernel_stability_path = output_dir / "kernel_shap_stability.csv"
    kernel_candidate_path = output_dir / "kernel_shap_candidate_summary.csv"
    kernel_stability.to_csv(kernel_stability_path, index=False)
    kernel_candidates.to_csv(kernel_candidate_path, index=False)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "conditional-within-cell-analysis",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "results_schema_version": schema_version,
                "results_csv_sha256": sha256_file(results_dir / "results.csv"),
                "bootstrap_repeats": bootstrap_repeats,
                "permutation_repeats": permutation_repeats,
                "seed": seed,
                "bootstrap_rows": len(bootstrap_rows),
                "lime_test_rows": len(lime_rows),
                "kernel_stability_rows": len(kernel_stability),
                "kernel_candidate_rows": len(kernel_candidates),
                "scope_note": (
                    "Intervals and paired sign-flip p-values are conditional within-cell "
                    "summaries; they are not independent-condition or population inference."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return AnalysisArtifacts(
        output_dir=output_dir,
        bootstrap_path=bootstrap_path,
        lime_test_path=lime_test_path,
        kernel_stability_path=kernel_stability_path,
        kernel_candidate_path=kernel_candidate_path,
        manifest_path=manifest_path,
        bootstrap_rows=len(bootstrap_rows),
        lime_test_rows=len(lime_rows),
        kernel_stability_rows=len(kernel_stability),
        kernel_candidate_rows=len(kernel_candidates),
    )
