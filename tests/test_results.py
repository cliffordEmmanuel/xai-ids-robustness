"""Tests for versioned result manifests and their validator."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from xai_ids.cli import main
from xai_ids.results import (
    RESULTS_SCHEMA_VERSION,
    validate_results_dir,
    write_run_manifest,
)
from xai_ids.runner import append_results_row


def _row(experiment_id: str = "unsw-xgboost-shap-tamp-pgd-e0.03-s42") -> dict:
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "results_version": "v4",
        "experiment_id": experiment_id,
        "dataset": "unsw_nb15",
        "model": "xgboost",
        "explainer": "shap",
        "explainer_backend": "tree",
        "shap_background_size": 100,
        "shap_kernel_nsamples": 100,
        "shap_kernel_l1_reg": "not_applicable",
        "shap_kernel_stability_repeats": 0,
        "shap_kernel_additivity_tolerance": 0.0,
        "shap_kernel_repeat_cosine_drift_tolerance": 0.0,
        "require_shap_kernel_stability": False,
        "lime_num_samples": 1500,
        "strand": "tamper",
        "method": "pgd",
        "objective": "attribution",
        "target_rule": "top1_suppress",
        "target_objective": "suppression",
        "target_topk": 10,
        "gradient_source": "spsa",
        "threat_model_model_access": "probability_query",
        "threat_model_explanation_access": "oracle",
        "model_query_budget": 125000,
        "explanation_query_budget": 250,
        "noise_floor_mode": "not_applicable",
        "noise_floor_repeats": 0,
        "epsilon": 0.03,
        "seed": 42,
        "n_total": 1,
        "n_preserved": 1,
        "n_flipped": 0,
        "retention_rate": 1.0,
        "n_frozen_due_prediction_flip": 0,
        "all_n_evaluated": 1,
        "preserved_n_evaluated": 1,
        "model_queries": 4,
        "explanation_queries": 2,
        "n_perturbed": 1,
        "model_accuracy": 1.0,
        "model_precision": 1.0,
        "model_recall": 1.0,
        "model_f1": 1.0,
        "model_test_n_total": 2,
        "model_test_n_benign": 1,
        "model_test_n_attack": 1,
        "model_true_negative": 1,
        "model_false_positive": 0,
        "model_false_negative": 0,
        "model_true_positive": 1,
        "model_min_attack_recall": 0.8,
        "model_min_f1": 0.8,
        "model_comparison_eligible": True,
        "n_explainer_convergence_warnings": 0,
        "shap_kernel_attributions_finite": True,
        "shap_kernel_max_additivity_error": 0.0,
        "shap_kernel_repeat_cosine_drift_mean": 0.0,
        "shap_kernel_repeat_cosine_drift_max": 0.0,
        "shap_kernel_stability_status": "not_applicable",
        "all_tamper_success_topk_displacement": 0.0,
        "preserved_tamper_success_topk_displacement": 0.0,
        "all_target_baseline_abs_attribution": 1.0,
        "preserved_target_baseline_abs_attribution": 1.0,
        "all_target_adversarial_abs_attribution": 0.5,
        "preserved_target_adversarial_abs_attribution": 0.5,
        "all_target_relative_attribution_reduction": 0.5,
        "preserved_target_relative_attribution_reduction": 0.5,
        "all_target_baseline_rank": 1.0,
        "preserved_target_baseline_rank": 1.0,
        "all_target_adversarial_rank": 1.0,
        "preserved_target_adversarial_rank": 1.0,
        "all_target_rank_deterioration": 0.0,
        "preserved_target_rank_deterioration": 0.0,
        "all_target_in_topk_baseline": 1.0,
        "preserved_target_in_topk_baseline": 1.0,
        "all_target_in_topk_adversarial": 1.0,
        "preserved_target_in_topk_adversarial": 1.0,
    }


def _complete_result(tmp_path, row: dict | None = None) -> dict:
    row = _row() if row is None else row
    run_dir = tmp_path / row["experiment_id"]
    run_dir.mkdir()
    for name in ("metrics.json", "provenance.json", "fidelity.json"):
        (run_dir / name).write_text(name, encoding="utf-8")
    pd.DataFrame(
        {
            "schema_version": [RESULTS_SCHEMA_VERSION],
            "results_version": [row["results_version"]],
            "experiment_id": [row["experiment_id"]],
            "flow_index": [0],
            "test_index": [9],
            "reference_prediction": [1],
            "adversarial_prediction": [1],
            "prediction_preserved": [True],
            "frozen_due_prediction_flip": [False],
            "perturbation_linf": [0.03],
            "perturbation_l1": [0.03],
            "perturbation_l2": [0.03],
            "n_changed_features": [1],
            "target_feature": [0],
            "target_applicable": [True],
            "target_objective": [row["target_objective"]],
            "target_topk": [row["target_topk"]],
            "target_baseline_abs_attribution": [1.0],
            "target_adversarial_abs_attribution": [0.5],
            "target_relative_attribution_reduction": [0.5],
            "target_baseline_rank": [1],
            "target_adversarial_rank": [1],
            "target_rank_deterioration": [0],
            "target_in_topk_baseline": [True],
            "target_in_topk_adversarial": [True],
            "target_success_topk_displacement": [False],
        }
    ).to_csv(run_dir / "per_flow.csv", index=False)
    np.savez_compressed(
        run_dir / "attack.npz",
        targets=np.array([0], dtype=int),
        objective_baseline_attributions=np.array([[1.0]]),
        adversarial_attributions=np.array([[0.5]]),
        target_objective=np.asarray(row["target_objective"]),
        target_topk=np.asarray(row["target_topk"], dtype=int),
    )
    np.savez_compressed(
        run_dir / "noise_floor.npz",
        mode=np.asarray(row["noise_floor_mode"]),
        seeds=np.array([], dtype=int),
        benign_drift_cosine=np.empty((0, 1)),
        benign_drift_l1=np.empty((0, 1)),
        benign_drift_l2=np.empty((0, 1)),
    )
    np.savez_compressed(
        run_dir / "kernel_shap_stability.npz",
        mode=np.asarray("not_applicable"),
        seeds=np.array([], dtype=int),
        repeat_cosine_drift=np.empty((0, 1)),
    )
    (run_dir / "explainer_diagnostics.json").write_text(
        json.dumps(
            {
                "kernel_l1_reg": row["shap_kernel_l1_reg"],
                "convergence_warning_count": row["n_explainer_convergence_warnings"],
                "summary": {
                    "shap_kernel_attributions_finite": row[
                        "shap_kernel_attributions_finite"
                    ],
                    "shap_kernel_max_additivity_error": row[
                        "shap_kernel_max_additivity_error"
                    ],
                    "shap_kernel_repeat_cosine_drift_mean": row[
                        "shap_kernel_repeat_cosine_drift_mean"
                    ],
                    "shap_kernel_repeat_cosine_drift_max": row[
                        "shap_kernel_repeat_cosine_drift_max"
                    ],
                    "shap_kernel_stability_status": row["shap_kernel_stability_status"],
                },
            }
        ),
        encoding="utf-8",
    )
    write_run_manifest(
        run_dir,
        row,
        artifact_names=(
            "fidelity.json",
            "per_flow.csv",
            "noise_floor.npz",
            "kernel_shap_stability.npz",
            "explainer_diagnostics.json",
        ),
    )
    append_results_row(tmp_path, row)
    return row


def test_validate_versioned_results_success(tmp_path):
    row = _complete_result(tmp_path)
    report = validate_results_dir(tmp_path, expected_ids=[row["experiment_id"]])
    assert report.ok
    assert report.rows == 1


def test_validate_results_rejects_legacy_shape(tmp_path):
    (tmp_path / "results.csv").write_text(
        "experiment_id,dataset\nunsw-xgboost-shap-tamp-e0.03-s42,unsw_nb15\n",
        encoding="utf-8",
    )
    report = validate_results_dir(tmp_path)
    assert not report.ok
    assert any("missing required columns" in error for error in report.errors)
    assert any("invalid or legacy experiment_id" in error for error in report.errors)


def test_validate_results_detects_artifact_tampering(tmp_path):
    _complete_result(tmp_path)
    run_dir = tmp_path / "unsw-xgboost-shap-tamp-pgd-e0.03-s42"
    (run_dir / "metrics.json").write_text("changed", encoding="utf-8")
    report = validate_results_dir(tmp_path)
    assert not report.ok
    assert any("checksum mismatch for metrics.json" in error for error in report.errors)


def test_validate_results_cli_reports_success(tmp_path, capsys):
    _complete_result(tmp_path)
    assert main(["validate-results", str(tmp_path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_results_rejects_required_kernel_shap_stability_failure(tmp_path):
    row = _row()
    row.update(
        {
            "explainer_backend": "kernel",
            "shap_kernel_l1_reg": "0p01",
            "shap_kernel_stability_repeats": 2,
            "shap_kernel_additivity_tolerance": 1e-6,
            "shap_kernel_repeat_cosine_drift_tolerance": 0.05,
            "require_shap_kernel_stability": True,
            "shap_kernel_repeat_cosine_drift_mean": 0.10,
            "shap_kernel_repeat_cosine_drift_max": 0.10,
            "shap_kernel_stability_status": "fail",
        }
    )
    _complete_result(tmp_path, row)
    run_dir = tmp_path / row["experiment_id"]
    np.savez_compressed(
        run_dir / "kernel_shap_stability.npz",
        mode=np.asarray("repeated_kernel_shap"),
        seeds=np.array([11, 13], dtype=int),
        repeat_cosine_drift=np.array([[0.10], [0.10]]),
    )
    write_run_manifest(
        run_dir,
        row,
        artifact_names=(
            "fidelity.json",
            "per_flow.csv",
            "noise_floor.npz",
            "kernel_shap_stability.npz",
            "explainer_diagnostics.json",
        ),
    )

    report = validate_results_dir(tmp_path)
    assert not report.ok
    assert any(
        "required KernelSHAP numerical-stability check failed" in error
        for error in report.errors
    )
    assert any("KernelSHAP repeat drift exceeds tolerance" in error for error in report.errors)


def test_run_manifest_records_artifact_checksums(tmp_path):
    row = _complete_result(tmp_path)
    manifest = json.loads(
        (tmp_path / row["experiment_id"] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == RESULTS_SCHEMA_VERSION
    assert set(manifest["artifacts"]) == {
        "attack.npz",
        "metrics.json",
        "provenance.json",
        "fidelity.json",
        "per_flow.csv",
        "noise_floor.npz",
        "kernel_shap_stability.npz",
        "explainer_diagnostics.json",
    }


def test_validate_results_rejects_per_flow_summary_disagreement(tmp_path):
    _complete_result(tmp_path)
    run_dir = tmp_path / "unsw-xgboost-shap-tamp-pgd-e0.03-s42"
    flows = pd.read_csv(run_dir / "per_flow.csv")
    flows.loc[0, "adversarial_prediction"] = 0
    flows.to_csv(run_dir / "per_flow.csv", index=False)
    report = validate_results_dir(tmp_path)
    assert not report.ok
    assert any("checksum mismatch for per_flow.csv" in error for error in report.errors)
    assert any("preservation flags disagree" in error for error in report.errors)


def test_validate_results_rejects_tamper_success_that_does_not_leave_topk(tmp_path):
    row = _complete_result(tmp_path)
    run_dir = tmp_path / row["experiment_id"]
    flows = pd.read_csv(run_dir / "per_flow.csv")
    flows.loc[0, "target_success_topk_displacement"] = True
    flows.to_csv(run_dir / "per_flow.csv", index=False)
    write_run_manifest(
        run_dir,
        row,
        artifact_names=(
            "fidelity.json",
            "per_flow.csv",
            "noise_floor.npz",
            "kernel_shap_stability.npz",
            "explainer_diagnostics.json",
        ),
    )

    report = validate_results_dir(tmp_path)
    assert not report.ok
    assert any(
        "tamper success does not satisfy target_rank > target_topk" in error
        for error in report.errors
    )


def test_validate_results_reconciles_tamper_outcomes_with_attack_artifact(tmp_path):
    row = _complete_result(tmp_path)
    run_dir = tmp_path / row["experiment_id"]
    flows = pd.read_csv(run_dir / "per_flow.csv")
    flows.loc[0, "target_adversarial_abs_attribution"] = 0.7
    flows.to_csv(run_dir / "per_flow.csv", index=False)
    write_run_manifest(
        run_dir,
        row,
        artifact_names=(
            "fidelity.json",
            "per_flow.csv",
            "noise_floor.npz",
            "kernel_shap_stability.npz",
            "explainer_diagnostics.json",
        ),
    )

    report = validate_results_dir(tmp_path)
    assert not report.ok
    assert any("does not match attack.npz evidence" in error for error in report.errors)


def test_validate_results_rejects_model_quality_that_disagrees_with_counts(tmp_path):
    row = _complete_result(tmp_path)
    row["model_f1"] = 0.5
    append_results_row(tmp_path, row)

    report = validate_results_dir(tmp_path)
    assert not report.ok
    assert any("model_f1 does not match its confusion counts" in error for error in report.errors)
