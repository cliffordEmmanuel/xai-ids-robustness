"""Tests for Phase 8 conditional within-cell analysis artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xai_ids.analysis import (
    _kernel_stability_reports,
    analyze_results_dir,
    bootstrap_mean_ci,
    paired_sign_flip_test,
    summarize_experiment,
)
from xai_ids.results import ResultsValidation


def _write_lime_run(tmp_path):
    run_dir = tmp_path / "cicids-xgboost-lime-frag-pgd-e0.03-s42"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "prediction_preserved": [True, False],
            "perturbation_linf": [0.01, 0.03],
            "perturbation_l1": [0.02, 0.04],
            "perturbation_l2": [0.02, 0.04],
        }
    ).to_csv(run_dir / "per_flow.csv", index=False)
    np.savez_compressed(
        run_dir / "attack.npz",
        baseline_attributions=np.array([[1.0, 0.0], [0.0, 1.0]]),
        adversarial_attributions=np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    np.savez_compressed(
        run_dir / "noise_floor.npz",
        mode=np.asarray("repeated_benign_lime"),
        benign_drift_cosine=np.full((2, 2), 0.1),
        benign_drift_l1=np.full((2, 2), 0.1),
        benign_drift_l2=np.full((2, 2), 0.1),
    )
    return run_dir


def _row(experiment_id: str) -> pd.Series:
    return pd.Series(
        {
            "schema_version": "3.5",
            "experiment_id": experiment_id,
            "dataset": "cicids2017",
            "model": "xgboost",
            "explainer": "lime",
            "explainer_backend": "lime",
            "strand": "fragility",
            "method": "pgd",
            "epsilon": 0.03,
            "seed": 42,
        }
    )


def test_bootstrap_interval_and_paired_test_are_seeded_and_conditional():
    values = np.array([0.0, 1.0, 2.0])
    estimate, low, high, n = bootstrap_mean_ci(
        values, repeats=100, rng=np.random.default_rng(3)
    )
    assert estimate == 1.0
    assert low <= estimate <= high
    assert n == 3

    effect, p_greater, n = paired_sign_flip_test(
        np.ones(20), repeats=500, rng=np.random.default_rng(4)
    )
    assert effect == 1.0
    assert 0.0 < p_greater < 0.05
    assert n == 20


def test_summarize_experiment_writes_bootstrap_and_lime_pairing_inputs(tmp_path):
    run_dir = _write_lime_run(tmp_path)
    bootstrap, lime = summarize_experiment(
        _row(run_dir.name),
        run_dir,
        bootstrap_repeats=100,
        permutation_repeats=100,
        rng=np.random.default_rng(5),
    )

    assert any(row["metric"] == "attack_drift_cosine" for row in bootstrap)
    assert any(row["cohort"] == "preserved" for row in bootstrap)
    assert {row["metric"] for row in lime} == {"cosine", "l1", "l2"}
    assert all(0.0 <= row["p_greater_sign_flip"] <= 1.0 for row in lime)


def test_analyze_results_dir_writes_versioned_analysis_artifacts(tmp_path, monkeypatch):
    run_dir = _write_lime_run(tmp_path)
    pd.DataFrame([_row(run_dir.name)]).to_csv(tmp_path / "results.csv", index=False)
    monkeypatch.setattr(
        "xai_ids.analysis.validate_results_dir",
        lambda _: ResultsValidation(errors=[], warnings=[], rows=1),
    )

    artifacts = analyze_results_dir(
        tmp_path,
        bootstrap_repeats=20,
        permutation_repeats=20,
        seed=6,
    )

    assert artifacts.bootstrap_path.is_file()
    assert artifacts.lime_test_path.is_file()
    assert artifacts.kernel_stability_path.is_file()
    assert artifacts.kernel_candidate_path.is_file()
    assert artifacts.manifest_path.is_file()
    assert artifacts.bootstrap_rows > 0
    assert artifacts.lime_test_rows == 6  # three drifts, all + preserved cohorts


def test_kernel_shap_candidate_summary_uses_only_numerical_release_checks():
    rows = pd.DataFrame(
        {
            "experiment_id": ["a", "b"],
            "dataset": ["cicids2017", "unsw_nb15"],
            "model": ["mlp", "xgboost"],
            "explainer": ["shap", "shap"],
            "explainer_backend": ["kernel", "kernel"],
            "strand": ["fragility", "tamper"],
            "method": ["pgd", "pgd"],
            "epsilon": [0.03, 0.03],
            "seed": [42, 42],
            "shap_kernel_nsamples": [100, 100],
            "shap_kernel_l1_reg": ["0p0001", "0p0001"],
            "shap_kernel_stability_repeats": [3, 3],
            "shap_kernel_additivity_tolerance": [1e-6, 1e-6],
            "shap_kernel_repeat_cosine_drift_tolerance": [0.05, 0.05],
            "require_shap_kernel_stability": [False, False],
            "n_explainer_convergence_warnings": [0, 1],
            "shap_kernel_attributions_finite": [True, True],
            "shap_kernel_max_additivity_error": [0.0, 0.0],
            "shap_kernel_repeat_cosine_drift_mean": [0.01, 0.01],
            "shap_kernel_repeat_cosine_drift_max": [0.01, 0.01],
            "shap_kernel_stability_status": ["pass", "fail"],
            # This deliberately favourable attack outcome must not enter the
            # numerical candidate-selection rule.
            "preserved_tamper_success_topk_displacement": [0.0, 1.0],
        }
    )
    per_cell, candidates = _kernel_stability_reports(rows)

    assert per_cell["cell_passes_numerical_rule"].tolist() == [True, False]
    assert len(candidates) == 1
    assert candidates.loc[0, "n_passing_cells"] == 1
    assert not candidates.loc[0, "all_cells_pass"]
