"""Versioned result manifests and validation for final experiment records."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_SCHEMA_VERSION = "3.6"
RESULTS_FILE = "results.csv"
ROOT_MANIFEST = "manifest.json"
RUN_MANIFEST = "manifest.json"
_IDENTITY_COLUMNS = frozenset(
    {
        "schema_version",
        "results_version",
        "experiment_id",
        "dataset",
        "model",
        "explainer",
        "strand",
        "method",
        "objective",
        "gradient_source",
        "epsilon",
        "seed",
    }
)

_PHASE2_SUMMARY_COLUMNS = frozenset(
    {
        "n_total",
        "n_preserved",
        "n_flipped",
        "retention_rate",
        "n_frozen_due_prediction_flip",
        "all_n_evaluated",
        "preserved_n_evaluated",
        "model_queries",
        "n_perturbed",
    }
)

# Schema 2 is retained solely to validate the Phase 0 verification record in
# results/v2. Schema 3.0 introduced explicit cohort reporting; schema 3.1 adds
# the declared oracle access and separates explanation-service requests from
# direct model probability queries; schema 3.2 adds repeated benign LIME noise
# artifacts and their reporting mode; schema 3.3 adds exact tamper target/rank
# evidence and the optional rank-margin objective; schema 3.4 adds frozen-test
# model quality and class-count evidence plus an explainer backend identity;
# schema 3.5 adds auditable explanation-time convergence-warning counts; schema
# 3.6 adds explicit KernelSHAP regularisation, repeated-stability evidence, and
# an enforceable numerical-stability release guard.
REQUIRED_RESULT_COLUMNS_BY_SCHEMA = {
    "2.0": _IDENTITY_COLUMNS
    | frozenset(
        {
            "n_evaluated",
            "retention_rate",
            "model_queries",
            "explainer_calls",
            "n_perturbed",
        }
    ),
    "3.0": _IDENTITY_COLUMNS | _PHASE2_SUMMARY_COLUMNS | {"explainer_calls"},
    "3.1": _IDENTITY_COLUMNS
    | _PHASE2_SUMMARY_COLUMNS
    | frozenset(
        {
            "threat_model_model_access",
            "threat_model_explanation_access",
            "model_query_budget",
            "explanation_query_budget",
            "explanation_queries",
        }
    ),
    "3.2": _IDENTITY_COLUMNS
    | _PHASE2_SUMMARY_COLUMNS
    | frozenset(
        {
            "threat_model_model_access",
            "threat_model_explanation_access",
            "model_query_budget",
            "explanation_query_budget",
            "explanation_queries",
            "noise_floor_mode",
            "noise_floor_repeats",
            "lime_num_samples",
        }
    ),
    "3.3": _IDENTITY_COLUMNS
    | _PHASE2_SUMMARY_COLUMNS
    | frozenset(
        {
            "threat_model_model_access",
            "threat_model_explanation_access",
            "model_query_budget",
            "explanation_query_budget",
            "explanation_queries",
            "noise_floor_mode",
            "noise_floor_repeats",
            "lime_num_samples",
            "target_rule",
            "target_objective",
            "target_topk",
        }
    ),
    "3.4": _IDENTITY_COLUMNS
    | _PHASE2_SUMMARY_COLUMNS
    | frozenset(
        {
            "threat_model_model_access",
            "threat_model_explanation_access",
            "model_query_budget",
            "explanation_query_budget",
            "explanation_queries",
            "noise_floor_mode",
            "noise_floor_repeats",
            "lime_num_samples",
            "explainer_backend",
            "shap_background_size",
            "shap_kernel_nsamples",
            "target_rule",
            "target_objective",
            "target_topk",
            "model_accuracy",
            "model_precision",
            "model_recall",
            "model_f1",
            "model_test_n_total",
            "model_test_n_benign",
            "model_test_n_attack",
            "model_true_negative",
            "model_false_positive",
            "model_false_negative",
            "model_true_positive",
            "model_min_attack_recall",
            "model_min_f1",
            "model_comparison_eligible",
        }
    ),
    "3.5": _IDENTITY_COLUMNS
    | _PHASE2_SUMMARY_COLUMNS
    | frozenset(
        {
            "threat_model_model_access",
            "threat_model_explanation_access",
            "model_query_budget",
            "explanation_query_budget",
            "explanation_queries",
            "noise_floor_mode",
            "noise_floor_repeats",
            "lime_num_samples",
            "explainer_backend",
            "shap_background_size",
            "shap_kernel_nsamples",
            "target_rule",
            "target_objective",
            "target_topk",
            "model_accuracy",
            "model_precision",
            "model_recall",
            "model_f1",
            "model_test_n_total",
            "model_test_n_benign",
            "model_test_n_attack",
            "model_true_negative",
            "model_false_positive",
            "model_false_negative",
            "model_true_positive",
            "model_min_attack_recall",
            "model_min_f1",
            "model_comparison_eligible",
            "n_explainer_convergence_warnings",
        }
    ),
}
REQUIRED_RESULT_COLUMNS_BY_SCHEMA["3.6"] = REQUIRED_RESULT_COLUMNS_BY_SCHEMA["3.5"] | frozenset(
    {
        "shap_kernel_l1_reg",
        "shap_kernel_stability_repeats",
        "shap_kernel_additivity_tolerance",
        "shap_kernel_repeat_cosine_drift_tolerance",
        "require_shap_kernel_stability",
        "shap_kernel_attributions_finite",
        "shap_kernel_max_additivity_error",
        "shap_kernel_repeat_cosine_drift_mean",
        "shap_kernel_repeat_cosine_drift_max",
        "shap_kernel_stability_status",
    }
)
REQUIRED_RESULT_COLUMNS = REQUIRED_RESULT_COLUMNS_BY_SCHEMA[RESULTS_SCHEMA_VERSION]

REQUIRED_RUN_ARTIFACTS_BY_SCHEMA = {
    "2.0": ("attack.npz", "metrics.json", "provenance.json", RUN_MANIFEST),
    "3.0": (
        "attack.npz",
        "metrics.json",
        "provenance.json",
        "fidelity.json",
        "per_flow.csv",
        RUN_MANIFEST,
    ),
    "3.1": (
        "attack.npz",
        "metrics.json",
        "provenance.json",
        "fidelity.json",
        "per_flow.csv",
        RUN_MANIFEST,
    ),
    "3.2": (
        "attack.npz",
        "metrics.json",
        "provenance.json",
        "fidelity.json",
        "per_flow.csv",
        "noise_floor.npz",
        RUN_MANIFEST,
    ),
    "3.3": (
        "attack.npz",
        "metrics.json",
        "provenance.json",
        "fidelity.json",
        "per_flow.csv",
        "noise_floor.npz",
        RUN_MANIFEST,
    ),
    "3.4": (
        "attack.npz",
        "metrics.json",
        "provenance.json",
        "fidelity.json",
        "per_flow.csv",
        "noise_floor.npz",
        RUN_MANIFEST,
    ),
    "3.5": (
        "attack.npz",
        "metrics.json",
        "provenance.json",
        "fidelity.json",
        "per_flow.csv",
        "noise_floor.npz",
        RUN_MANIFEST,
    ),
}
REQUIRED_RUN_ARTIFACTS_BY_SCHEMA["3.6"] = (
    *REQUIRED_RUN_ARTIFACTS_BY_SCHEMA["3.5"][:-1],
    "kernel_shap_stability.npz",
    "explainer_diagnostics.json",
    RUN_MANIFEST,
)
REQUIRED_RUN_ARTIFACTS = REQUIRED_RUN_ARTIFACTS_BY_SCHEMA[RESULTS_SCHEMA_VERSION]

_BASE_PER_FLOW_COLUMNS = frozenset(
    {
        "schema_version",
        "results_version",
        "experiment_id",
        "flow_index",
        "test_index",
        "reference_prediction",
        "adversarial_prediction",
        "prediction_preserved",
        "frozen_due_prediction_flip",
        "perturbation_linf",
        "perturbation_l1",
        "perturbation_l2",
        "n_changed_features",
        "target_feature",
    }
)
_TAMPER_OUTCOME_COLUMNS = frozenset(
    {
        "target_applicable",
        "target_objective",
        "target_topk",
        "target_baseline_abs_attribution",
        "target_adversarial_abs_attribution",
        "target_relative_attribution_reduction",
        "target_baseline_rank",
        "target_adversarial_rank",
        "target_rank_deterioration",
        "target_in_topk_baseline",
        "target_in_topk_adversarial",
        "target_success_topk_displacement",
    }
)
REQUIRED_PER_FLOW_COLUMNS_BY_SCHEMA = {
    "3.0": _BASE_PER_FLOW_COLUMNS,
    "3.1": _BASE_PER_FLOW_COLUMNS,
    "3.2": _BASE_PER_FLOW_COLUMNS,
    "3.3": _BASE_PER_FLOW_COLUMNS | _TAMPER_OUTCOME_COLUMNS,
    "3.4": _BASE_PER_FLOW_COLUMNS | _TAMPER_OUTCOME_COLUMNS,
    "3.5": _BASE_PER_FLOW_COLUMNS | _TAMPER_OUTCOME_COLUMNS,
    "3.6": _BASE_PER_FLOW_COLUMNS | _TAMPER_OUTCOME_COLUMNS,
}
REQUIRED_PER_FLOW_COLUMNS = REQUIRED_PER_FLOW_COLUMNS_BY_SCHEMA[RESULTS_SCHEMA_VERSION]
EXPERIMENT_ID_RE = re.compile(r".+-(?:fgsm|pgd)-e[^-]+-s\d+$")


@dataclass(frozen=True)
class ResultsValidation:
    """Result of validating a versioned results directory."""

    errors: list[str]
    warnings: list[str]
    rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_run_manifest(
    run_dir: str | Path, row: dict, *, artifact_names: tuple[str, ...] = ()
) -> Path:
    """Write checksums and row identity for one completed experiment artifact set."""
    run_dir = Path(run_dir)
    artifact_names = ("attack.npz", "metrics.json", "provenance.json", *artifact_names)
    artifacts = {
        name: {"sha256": sha256_file(run_dir / name)}
        for name in artifact_names
        if (run_dir / name).is_file()
    }
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "kind": "experiment-run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": row["experiment_id"],
        "results_version": row["results_version"],
        "artifacts": artifacts,
    }
    return _write_json(run_dir / RUN_MANIFEST, payload)


def write_results_manifest(output_dir: str | Path, frame: pd.DataFrame) -> Path:
    """Write the compact, versioned manifest for a complete/in-progress results table."""
    output_dir = Path(output_dir)
    results_path = output_dir / RESULTS_FILE
    versions = sorted({str(value) for value in frame.get("schema_version", pd.Series())})
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "kind": "results-table",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "results_file": RESULTS_FILE,
        "results_sha256": sha256_file(results_path),
        "row_schema_versions": versions,
    }
    return _write_json(output_dir / ROOT_MANIFEST, payload)


def validate_results_dir(
    output_dir: str | Path, *, expected_ids: Iterable[str] | None = None
) -> ResultsValidation:
    """Validate result-table structure, manifests, artifacts, and optional grid coverage."""
    output_dir = Path(output_dir)
    errors: list[str] = []
    warnings: list[str] = []
    results_path = output_dir / RESULTS_FILE
    if not results_path.is_file():
        return ResultsValidation([f"missing results table: {results_path}"], warnings)

    try:
        frame = pd.read_csv(results_path)
    except Exception as exc:  # noqa: BLE001 - present a single actionable validation error
        return ResultsValidation([f"cannot read {results_path}: {exc}"], warnings)

    if frame.empty:
        errors.append("results table contains no rows")
        return ResultsValidation(errors, warnings, rows=0)

    row_schemas: list[str] = []
    if "schema_version" not in frame:
        errors.append("missing required columns: schema_version")
    elif frame["schema_version"].isna().any():
        errors.append("null required values in columns: schema_version")
    else:
        row_schemas = sorted(set(frame["schema_version"].astype(str)))
        if len(row_schemas) != 1:
            errors.append("results table must use exactly one schema_version")
        elif row_schemas[0] not in REQUIRED_RESULT_COLUMNS_BY_SCHEMA:
            errors.append(f"unsupported schema_version: {row_schemas[0]}")

    schema_version = row_schemas[0] if len(row_schemas) == 1 else None
    required_columns = REQUIRED_RESULT_COLUMNS_BY_SCHEMA.get(schema_version, frozenset())
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        errors.append(f"missing required columns: {', '.join(missing_columns)}")

    if "experiment_id" in frame:
        duplicated = frame["experiment_id"].duplicated(keep=False)
        if duplicated.any():
            ids = sorted(frame.loc[duplicated, "experiment_id"].astype(str).unique())
            errors.append(f"duplicate experiment_id values: {', '.join(ids)}")
        invalid_ids = [
            value
            for value in frame["experiment_id"].astype(str)
            if not EXPERIMENT_ID_RE.fullmatch(value)
        ]
        if invalid_ids:
            invalid = ", ".join(sorted(set(invalid_ids)))
            errors.append(f"invalid or legacy experiment_id values: {invalid}")

    present_required = sorted(required_columns & set(frame.columns))
    if present_required:
        nulls = frame[present_required].isna().any()
        null_columns = nulls[nulls].index.tolist()
        if null_columns:
            errors.append(f"null required values in columns: {', '.join(null_columns)}")

    manifest_path = output_dir / ROOT_MANIFEST
    if not manifest_path.is_file():
        errors.append(f"missing results manifest: {manifest_path}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != schema_version:
                errors.append("results manifest schema version does not match")
            if manifest.get("results_sha256") != sha256_file(results_path):
                errors.append("results manifest checksum does not match results.csv")
            if manifest.get("rows") != len(frame):
                errors.append("results manifest row count does not match results.csv")
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"cannot read results manifest: {exc}")

    if "experiment_id" in frame:
        for experiment_id in frame["experiment_id"].astype(str):
            run_dir = output_dir / experiment_id
            required_artifacts = REQUIRED_RUN_ARTIFACTS_BY_SCHEMA.get(schema_version, ())
            for artifact in required_artifacts:
                if not (run_dir / artifact).is_file():
                    errors.append(f"{experiment_id}: missing required artifact {artifact}")
            run_manifest_path = run_dir / RUN_MANIFEST
            if run_manifest_path.is_file():
                try:
                    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
                    if run_manifest.get("experiment_id") != experiment_id:
                        errors.append(f"{experiment_id}: run manifest experiment_id mismatch")
                    if run_manifest.get("schema_version") != schema_version:
                        errors.append(
                            f"{experiment_id}: run manifest schema version does not match"
                        )
                    recorded_artifacts = run_manifest.get("artifacts", {})
                    for artifact in required_artifacts:
                        if artifact != RUN_MANIFEST and artifact not in recorded_artifacts:
                            errors.append(
                                f"{experiment_id}: run manifest is missing checksum for {artifact}"
                            )
                    for name, metadata in recorded_artifacts.items():
                        artifact_path = run_dir / name
                        if artifact_path.is_file():
                            checksum_matches = metadata.get("sha256") == sha256_file(artifact_path)
                            if not checksum_matches:
                                errors.append(f"{experiment_id}: checksum mismatch for {name}")
                except (json.JSONDecodeError, OSError) as exc:
                    errors.append(f"{experiment_id}: cannot read run manifest: {exc}")

            if schema_version in {"3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6"}:
                _validate_per_flow(
                    run_dir / "per_flow.csv",
                    frame.loc[frame["experiment_id"].astype(str) == experiment_id].iloc[0],
                    errors,
                )
            if schema_version in {"3.1", "3.2", "3.3", "3.4", "3.5", "3.6"}:
                _validate_threat_model_summary(
                    frame.loc[frame["experiment_id"].astype(str) == experiment_id].iloc[0],
                    errors,
                )
            if schema_version in {"3.2", "3.3", "3.4", "3.5", "3.6"}:
                _validate_noise_floor_artifact(
                    run_dir / "noise_floor.npz",
                    frame.loc[frame["experiment_id"].astype(str) == experiment_id].iloc[0],
                    errors,
                )
            if schema_version in {"3.3", "3.4", "3.5", "3.6"}:
                _validate_tamper_attack_artifact(
                    run_dir / "attack.npz",
                    run_dir / "per_flow.csv",
                    frame.loc[frame["experiment_id"].astype(str) == experiment_id].iloc[0],
                    errors,
                )
            if schema_version in {"3.4", "3.5", "3.6"}:
                _validate_model_quality_summary(
                    frame.loc[frame["experiment_id"].astype(str) == experiment_id].iloc[0],
                    errors,
                )
            if schema_version in {"3.5", "3.6"}:
                _validate_explainer_warning_summary(
                    frame.loc[frame["experiment_id"].astype(str) == experiment_id].iloc[0],
                    errors,
                )
            if schema_version == "3.6":
                row = frame.loc[frame["experiment_id"].astype(str) == experiment_id].iloc[0]
                _validate_kernel_shap_stability_summary(row, errors)
                _validate_kernel_shap_stability_artifact(
                    run_dir / "kernel_shap_stability.npz", row, errors
                )
                _validate_explainer_diagnostics_artifact(
                    run_dir / "explainer_diagnostics.json", row, errors
                )

    if expected_ids is not None and "experiment_id" in frame:
        expected = set(expected_ids)
        observed = set(frame["experiment_id"].astype(str))
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing:
            errors.append(f"missing expected experiment IDs: {', '.join(missing)}")
        if unexpected:
            errors.append(f"unexpected experiment IDs: {', '.join(unexpected)}")

    return ResultsValidation(errors, warnings, rows=len(frame))


def _as_boolean(series: pd.Series) -> pd.Series | None:
    """Return a boolean series for CSV booleans, or ``None`` when malformed."""
    normalised = series.astype(str).str.lower()
    values = {"true": True, "false": False}
    if not normalised.isin(values).all():
        return None
    return normalised.map(values).astype(bool)


def _validate_per_flow(path: Path, row: pd.Series, errors: list[str]) -> None:
    """Validate the Phase 2 row-level attack evidence against its summary row."""
    if not path.is_file():
        return  # The required-artifact error above is clearer.
    try:
        flows = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - report a compact validation error
        errors.append(f"{row['experiment_id']}: cannot read per_flow.csv: {exc}")
        return

    required_columns = REQUIRED_PER_FLOW_COLUMNS_BY_SCHEMA.get(
        str(row["schema_version"]), frozenset()
    )
    missing_columns = sorted(required_columns - set(flows.columns))
    if missing_columns:
        errors.append(
            f"{row['experiment_id']}: per_flow.csv missing columns: "
            f"{', '.join(missing_columns)}"
        )
        return
    if len(flows) != int(row["n_total"]):
        errors.append(f"{row['experiment_id']}: per_flow.csv row count does not match n_total")
    if set(flows["experiment_id"].astype(str)) != {str(row["experiment_id"])}:
        errors.append(f"{row['experiment_id']}: per_flow.csv experiment_id mismatch")
    if set(flows["schema_version"].astype(str)) != {str(row["schema_version"])}:
        errors.append(f"{row['experiment_id']}: per_flow.csv schema_version mismatch")
    if set(flows["results_version"].astype(str)) != {str(row["results_version"])}:
        errors.append(f"{row['experiment_id']}: per_flow.csv results_version mismatch")

    preserved = _as_boolean(flows["prediction_preserved"])
    if preserved is None:
        errors.append(
            f"{row['experiment_id']}: invalid prediction_preserved values in per_flow.csv"
        )
        return
    prediction_matches = flows["reference_prediction"] == flows["adversarial_prediction"]
    if not preserved.equals(prediction_matches.astype(bool)):
        errors.append(f"{row['experiment_id']}: preservation flags disagree with predictions")
    if int(preserved.sum()) != int(row["n_preserved"]):
        errors.append(
            f"{row['experiment_id']}: per_flow.csv preserved count does not match n_preserved"
        )
    if int((~preserved).sum()) != int(row["n_flipped"]):
        errors.append(
            f"{row['experiment_id']}: per_flow.csv flipped count does not match n_flipped"
        )
    frozen = _as_boolean(flows["frozen_due_prediction_flip"])
    if frozen is None:
        errors.append(
            f"{row['experiment_id']}: invalid frozen_due_prediction_flip values in per_flow.csv"
        )
    elif int(frozen.sum()) != int(row["n_frozen_due_prediction_flip"]):
        errors.append(f"{row['experiment_id']}: frozen count does not match summary")
    if int((flows["n_changed_features"] > 0).sum()) != int(row["n_perturbed"]):
        errors.append(f"{row['experiment_id']}: perturbed count does not match summary")

    n_total = int(row["n_total"])
    n_preserved = int(row["n_preserved"])
    if int(row["n_flipped"]) + n_preserved != n_total:
        errors.append(f"{row['experiment_id']}: n_preserved and n_flipped do not sum to n_total")
    if int(row["all_n_evaluated"]) != n_total:
        errors.append(f"{row['experiment_id']}: all_n_evaluated must equal n_total")
    if int(row["preserved_n_evaluated"]) != n_preserved:
        errors.append(f"{row['experiment_id']}: preserved_n_evaluated must equal n_preserved")
    expected_retention = n_preserved / n_total if n_total else 0.0
    if not np.isclose(float(row["retention_rate"]), expected_retention):
        errors.append(f"{row['experiment_id']}: retention_rate does not match preserved count")
    if str(row["schema_version"]) in {"3.3", "3.4", "3.5", "3.6"}:
        _validate_tamper_outcomes(flows, row, preserved, errors)


def _validate_tamper_outcomes(
    flows: pd.DataFrame, row: pd.Series, preserved: pd.Series, errors: list[str]
) -> None:
    """Validate exact target rank/displacement evidence in a schema-3.3 run."""
    experiment_id = row["experiment_id"]
    applicable = _as_boolean(flows["target_applicable"])
    if applicable is None:
        errors.append(f"{experiment_id}: invalid target_applicable values in per_flow.csv")
        return
    is_tamper = row["strand"] == "tamper"
    if is_tamper and not applicable.all():
        errors.append(f"{experiment_id}: tamper rows require target evidence for every flow")
        return
    if not is_tamper:
        if applicable.any():
            errors.append(f"{experiment_id}: non-tamper rows must not contain target evidence")
        return

    target_topk = pd.to_numeric(flows["target_topk"], errors="coerce")
    if target_topk.isna().any() or not (target_topk == int(row["target_topk"])).all():
        errors.append(f"{experiment_id}: per-flow target_topk does not match result row")
    if set(flows["target_objective"].astype(str)) != {str(row["target_objective"])}:
        errors.append(f"{experiment_id}: per-flow target_objective does not match result row")

    baseline_rank = pd.to_numeric(flows["target_baseline_rank"], errors="coerce")
    adversarial_rank = pd.to_numeric(flows["target_adversarial_rank"], errors="coerce")
    if baseline_rank.isna().any() or adversarial_rank.isna().any():
        errors.append(f"{experiment_id}: target ranks contain missing values")
        return
    if (baseline_rank < 1).any() or (adversarial_rank < 1).any():
        errors.append(f"{experiment_id}: target ranks must be positive")
    if not np.equal(baseline_rank, np.floor(baseline_rank)).all() or not np.equal(
        adversarial_rank, np.floor(adversarial_rank)
    ).all():
        errors.append(f"{experiment_id}: target ranks must be integers")

    for column in (
        "target_baseline_abs_attribution",
        "target_adversarial_abs_attribution",
    ):
        values = pd.to_numeric(flows[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any() or (values < 0).any():
            errors.append(f"{experiment_id}: {column} must be finite and non-negative")
    baseline_abs = pd.to_numeric(flows["target_baseline_abs_attribution"], errors="coerce")
    adversarial_abs = pd.to_numeric(
        flows["target_adversarial_abs_attribution"], errors="coerce"
    )
    relative_reduction = pd.to_numeric(
        flows["target_relative_attribution_reduction"], errors="coerce"
    )
    expected_reduction = np.divide(
        baseline_abs - adversarial_abs,
        baseline_abs,
        out=np.full(len(flows), np.nan),
        where=baseline_abs > 0,
    )
    if not np.allclose(relative_reduction, expected_reduction, equal_nan=True):
        errors.append(f"{experiment_id}: target relative reduction does not match magnitudes")

    if row["target_rule"] == "top1_suppress" and not (baseline_rank == 1).all():
        errors.append(f"{experiment_id}: top-1 targets must have baseline rank one")
    rank_deterioration = pd.to_numeric(flows["target_rank_deterioration"], errors="coerce")
    if not np.allclose(rank_deterioration, adversarial_rank - baseline_rank, equal_nan=False):
        errors.append(f"{experiment_id}: target rank deterioration does not match ranks")

    baseline_membership = _as_boolean(flows["target_in_topk_baseline"])
    adversarial_membership = _as_boolean(flows["target_in_topk_adversarial"])
    success = _as_boolean(flows["target_success_topk_displacement"])
    if baseline_membership is None or adversarial_membership is None or success is None:
        errors.append(f"{experiment_id}: invalid target membership/success booleans")
        return
    topk = int(row["target_topk"])
    if not baseline_membership.equals((baseline_rank <= topk).astype(bool)):
        errors.append(f"{experiment_id}: baseline top-k membership does not match target rank")
    if not adversarial_membership.equals((adversarial_rank <= topk).astype(bool)):
        errors.append(f"{experiment_id}: adversarial top-k membership does not match target rank")
    if not success.equals((adversarial_rank > topk).astype(bool)):
        errors.append(f"{experiment_id}: tamper success does not satisfy target_rank > target_topk")

    _validate_tamper_summary(
        row,
        flows,
        success,
        preserved,
        errors,
    )


def _target_ranks(attributions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Rank targets by absolute attribution with deterministic tie handling."""
    order = np.argsort(-np.abs(attributions), axis=1, kind="stable")
    return (order == targets[:, None]).argmax(axis=1) + 1


def _validate_tamper_attack_artifact(
    attack_path: Path, per_flow_path: Path, row: pd.Series, errors: list[str]
) -> None:
    """Reconcile schema-3.3 per-flow outcomes with exact attack NPZ evidence."""
    if row["strand"] != "tamper" or not attack_path.is_file() or not per_flow_path.is_file():
        return
    experiment_id = row["experiment_id"]
    try:
        flows = pd.read_csv(per_flow_path)
        with np.load(attack_path, allow_pickle=False) as artifact:
            required = {
                "targets",
                "objective_baseline_attributions",
                "adversarial_attributions",
                "target_objective",
                "target_topk",
            }
            missing = sorted(required - set(artifact.files))
            if missing:
                errors.append(
                    f"{experiment_id}: attack.npz missing tamper evidence: {', '.join(missing)}"
                )
                return
            targets = np.asarray(artifact["targets"], dtype=int)
            baseline = np.asarray(artifact["objective_baseline_attributions"], dtype=float)
            adversarial = np.asarray(artifact["adversarial_attributions"], dtype=float)
            if (
                targets.shape != (len(flows),)
                or baseline.ndim != 2
                or baseline.shape != adversarial.shape
                or baseline.shape[0] != len(flows)
            ):
                errors.append(
                    f"{experiment_id}: attack.npz tamper evidence has incompatible shapes"
                )
                return
            if (targets < 0).any() or (targets >= baseline.shape[1]).any():
                errors.append(f"{experiment_id}: attack.npz target feature index is invalid")
                return
            if str(artifact["target_objective"].item()) != str(row["target_objective"]):
                errors.append(
                    f"{experiment_id}: attack.npz target_objective does not match result row"
                )
            if int(artifact["target_topk"].item()) != int(row["target_topk"]):
                errors.append(f"{experiment_id}: attack.npz target_topk does not match result row")

            rows = np.arange(len(flows))
            expected = {
                "target_feature": targets,
                "target_baseline_abs_attribution": np.abs(baseline[rows, targets]),
                "target_adversarial_abs_attribution": np.abs(adversarial[rows, targets]),
                "target_baseline_rank": _target_ranks(baseline, targets),
                "target_adversarial_rank": _target_ranks(adversarial, targets),
            }
            for column, values in expected.items():
                observed = pd.to_numeric(flows[column], errors="coerce").to_numpy()
                if not np.allclose(observed, values, equal_nan=False):
                    errors.append(f"{experiment_id}: {column} does not match attack.npz evidence")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"{experiment_id}: cannot validate tamper attack evidence: {exc}")


def _validate_tamper_summary(
    row: pd.Series,
    flows: pd.DataFrame,
    success: pd.Series,
    preserved: pd.Series,
    errors: list[str],
) -> None:
    """Reconcile all/preserved tamper summaries with per-flow target evidence."""
    experiment_id = row["experiment_id"]
    supporting_columns = (
        "target_baseline_abs_attribution",
        "target_adversarial_abs_attribution",
        "target_relative_attribution_reduction",
        "target_baseline_rank",
        "target_adversarial_rank",
        "target_rank_deterioration",
        "target_in_topk_baseline",
        "target_in_topk_adversarial",
    )
    for prefix, cohort in (("all_", np.ones(len(success), dtype=bool)), ("preserved_", preserved)):
        column = f"{prefix}tamper_success_topk_displacement"
        if not cohort.any():
            continue
        if column not in row.index or pd.isna(row[column]):
            errors.append(f"{experiment_id}: missing {column} summary")
            continue
        expected = float(success[cohort].mean())
        if not np.isclose(float(row[column]), expected):
            errors.append(f"{experiment_id}: {column} does not match per-flow success")
        for per_flow_column in supporting_columns:
            summary_column = f"{prefix}{per_flow_column}"
            if summary_column not in row.index:
                errors.append(f"{experiment_id}: missing {summary_column} summary")
                continue
            values = pd.to_numeric(flows[per_flow_column], errors="coerce").to_numpy()[cohort]
            finite = np.isfinite(values)
            expected_value = float(values[finite].mean()) if finite.any() else float("nan")
            observed = row[summary_column]
            if np.isnan(expected_value):
                if not pd.isna(observed):
                    errors.append(
                        f"{experiment_id}: {summary_column} must be null when no finite values "
                        "exist"
                    )
            elif pd.isna(observed) or not np.isclose(float(observed), expected_value):
                errors.append(
                    f"{experiment_id}: {summary_column} does not match per-flow target evidence"
                )


def _validate_threat_model_summary(row: pd.Series, errors: list[str]) -> None:
    """Check that a schema-3.1 run respects and names its declared access cap."""
    experiment_id = row["experiment_id"]
    if row["threat_model_model_access"] != "probability_query":
        errors.append(f"{experiment_id}: unsupported model access declaration")
    if row["threat_model_explanation_access"] not in {"oracle", "none"}:
        errors.append(f"{experiment_id}: unsupported explanation access declaration")
    if row["strand"] == "tamper" and row["threat_model_explanation_access"] != "oracle":
        errors.append(f"{experiment_id}: tamper result requires explanation-oracle access")
    if int(row["model_queries"]) > int(row["model_query_budget"]):
        errors.append(f"{experiment_id}: model-query budget exceeded in result row")
    if int(row["explanation_queries"]) > int(row["explanation_query_budget"]):
        errors.append(f"{experiment_id}: explanation-query budget exceeded in result row")


def _validate_model_quality_summary(row: pd.Series, errors: list[str]) -> None:
    """Check frozen-test model metrics, class counts, and comparison eligibility."""
    experiment_id = row["experiment_id"]
    backend = str(row["explainer_backend"])
    if row["explainer"] == "shap" and backend not in {"tree", "kernel"}:
        errors.append(f"{experiment_id}: SHAP result has an invalid explainer_backend")
    if row["explainer"] == "lime" and backend != "lime":
        errors.append(f"{experiment_id}: LIME result must use explainer_backend='lime'")
    if int(row["shap_background_size"]) < 1 or int(row["shap_kernel_nsamples"]) < 1:
        errors.append(f"{experiment_id}: SHAP sampling settings must be positive")

    score_names = ("model_accuracy", "model_precision", "model_recall", "model_f1")
    for name in score_names:
        value = float(row[name])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            errors.append(f"{experiment_id}: {name} must be between zero and one")

    count_names = (
        "model_test_n_total",
        "model_test_n_benign",
        "model_test_n_attack",
        "model_true_negative",
        "model_false_positive",
        "model_false_negative",
        "model_true_positive",
    )
    counts: dict[str, int] = {}
    for name in count_names:
        value = float(row[name])
        if not np.isfinite(value) or value < 0 or not value.is_integer():
            errors.append(f"{experiment_id}: {name} must be a non-negative integer")
        else:
            counts[name] = int(value)
    if len(counts) != len(count_names):
        return

    if counts["model_test_n_benign"] + counts["model_test_n_attack"] != counts[
        "model_test_n_total"
    ]:
        errors.append(f"{experiment_id}: test class counts do not sum to model_test_n_total")
    if counts["model_true_negative"] + counts["model_false_positive"] != counts[
        "model_test_n_benign"
    ]:
        errors.append(f"{experiment_id}: negative-class confusion counts do not match test count")
    if counts["model_false_negative"] + counts["model_true_positive"] != counts[
        "model_test_n_attack"
    ]:
        errors.append(f"{experiment_id}: attack-class confusion counts do not match test count")

    total = counts["model_test_n_total"]
    expected_accuracy = (
        (counts["model_true_negative"] + counts["model_true_positive"]) / total if total else 0.0
    )
    predicted_attack = counts["model_true_positive"] + counts["model_false_positive"]
    expected_precision = (
        counts["model_true_positive"] / predicted_attack if predicted_attack else 0.0
    )
    actual_attack = counts["model_test_n_attack"]
    expected_recall = counts["model_true_positive"] / actual_attack if actual_attack else 0.0
    expected_f1 = (
        2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0.0
    )
    expected_scores = {
        "model_accuracy": expected_accuracy,
        "model_precision": expected_precision,
        "model_recall": expected_recall,
        "model_f1": expected_f1,
    }
    for name, expected in expected_scores.items():
        if not np.isclose(float(row[name]), expected):
            errors.append(f"{experiment_id}: {name} does not match its confusion counts")

    min_recall = float(row["model_min_attack_recall"])
    min_f1 = float(row["model_min_f1"])
    if not np.isclose(min_recall, 0.80) or not np.isclose(min_f1, 0.80):
        errors.append(
            f"{experiment_id}: model-quality thresholds must use the declared 0.80 policy"
        )
    eligible = _as_boolean(pd.Series([row["model_comparison_eligible"]]))
    if eligible is None:
        errors.append(f"{experiment_id}: invalid model_comparison_eligible value")
    else:
        expected_eligible = (
            float(row["model_recall"]) >= min_recall and float(row["model_f1"]) >= min_f1
        )
        if bool(eligible.iloc[0]) != expected_eligible:
            errors.append(
                f"{experiment_id}: model_comparison_eligible disagrees with quality metrics"
            )


def _validate_explainer_warning_summary(row: pd.Series, errors: list[str]) -> None:
    """Require an auditable non-negative explanation-time warning count."""
    experiment_id = row["experiment_id"]
    if "n_explainer_convergence_warnings" not in row.index:
        return  # The required-column validator already emitted the clearer error.
    value = float(row["n_explainer_convergence_warnings"])
    if not np.isfinite(value) or value < 0 or not value.is_integer():
        errors.append(
            f"{experiment_id}: n_explainer_convergence_warnings must be a non-negative integer"
        )


def _as_nonnegative_integer(
    value, *, name: str, experiment_id: str, errors: list[str]
) -> int | None:
    """Parse a non-negative integer result field while retaining a compact error."""
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        errors.append(f"{experiment_id}: {name} must be a non-negative integer")
        return None
    return int(numeric)


def _validate_kernel_shap_stability_summary(row: pd.Series, errors: list[str]) -> None:
    """Validate schema-3.6 numerical-stability fields and final-release guard."""
    experiment_id = str(row["experiment_id"])
    kernel = row["explainer"] == "shap" and row["explainer_backend"] == "kernel"
    required = _as_boolean(pd.Series([row["require_shap_kernel_stability"]]))
    finite = _as_boolean(pd.Series([row["shap_kernel_attributions_finite"]]))
    if required is None:
        errors.append(f"{experiment_id}: invalid require_shap_kernel_stability value")
        return
    if finite is None:
        errors.append(f"{experiment_id}: invalid shap_kernel_attributions_finite value")
        return

    repeats = _as_nonnegative_integer(
        row["shap_kernel_stability_repeats"],
        name="shap_kernel_stability_repeats",
        experiment_id=experiment_id,
        errors=errors,
    )
    warning_count = _as_nonnegative_integer(
        row["n_explainer_convergence_warnings"],
        name="n_explainer_convergence_warnings",
        experiment_id=experiment_id,
        errors=errors,
    )
    numeric_names = (
        "shap_kernel_additivity_tolerance",
        "shap_kernel_repeat_cosine_drift_tolerance",
        "shap_kernel_max_additivity_error",
        "shap_kernel_repeat_cosine_drift_mean",
        "shap_kernel_repeat_cosine_drift_max",
    )
    numeric: dict[str, float] = {}
    for name in numeric_names:
        value = float(row[name])
        if not np.isfinite(value) or value < 0:
            errors.append(f"{experiment_id}: {name} must be a finite non-negative number")
        else:
            numeric[name] = value

    status = str(row["shap_kernel_stability_status"])
    l1_reg = str(row["shap_kernel_l1_reg"])
    if not kernel:
        if l1_reg != "not_applicable" or status != "not_applicable":
            errors.append(f"{experiment_id}: non-KernelSHAP rows must be marked not_applicable")
        if bool(required.iloc[0]) or repeats not in {0, None}:
            errors.append(f"{experiment_id}: non-KernelSHAP rows cannot require stability checks")
        return

    if l1_reg == "not_applicable" or not l1_reg:
        errors.append(f"{experiment_id}: KernelSHAP rows require an explicit l1 regularisation")
    if status not in {"not_checked", "pass", "fail"}:
        errors.append(f"{experiment_id}: invalid shap_kernel_stability_status")
    if repeats is not None and repeats and repeats < 2:
        errors.append(f"{experiment_id}: KernelSHAP stability repeats must be zero or at least two")
    if bool(required.iloc[0]):
        if repeats is None or repeats < 2:
            errors.append(
                f"{experiment_id}: required KernelSHAP stability needs at least two repeats"
            )
        if status != "pass":
            errors.append(f"{experiment_id}: required KernelSHAP numerical-stability check failed")
        if not bool(finite.iloc[0]):
            errors.append(
                f"{experiment_id}: required KernelSHAP result has non-finite attributions"
            )
        if warning_count not in {0, None}:
            errors.append(f"{experiment_id}: required KernelSHAP result has convergence warnings")
        if len(numeric) == len(numeric_names):
            if numeric["shap_kernel_max_additivity_error"] > numeric[
                "shap_kernel_additivity_tolerance"
            ]:
                errors.append(f"{experiment_id}: KernelSHAP additivity error exceeds tolerance")
            if numeric["shap_kernel_repeat_cosine_drift_max"] > numeric[
                "shap_kernel_repeat_cosine_drift_tolerance"
            ]:
                errors.append(f"{experiment_id}: KernelSHAP repeat drift exceeds tolerance")


def _validate_kernel_shap_stability_artifact(
    path: Path, row: pd.Series, errors: list[str]
) -> None:
    """Reconcile the repeat-drift evidence artifact with its schema-3.6 summary."""
    experiment_id = str(row["experiment_id"])
    if not path.is_file():
        return
    try:
        with np.load(path, allow_pickle=False) as artifact:
            required = {"mode", "seeds", "repeat_cosine_drift"}
            missing = sorted(required - set(artifact.files))
            if missing:
                errors.append(
                    f"{experiment_id}: kernel_shap_stability.npz missing fields: "
                    f"{', '.join(missing)}"
                )
                return
            mode = str(np.asarray(artifact["mode"]).item())
            seeds = np.asarray(artifact["seeds"], dtype=int)
            drift = np.asarray(artifact["repeat_cosine_drift"], dtype=float)
    except (OSError, ValueError) as exc:
        errors.append(f"{experiment_id}: cannot read kernel_shap_stability.npz: {exc}")
        return

    repeats = int(row["shap_kernel_stability_repeats"])
    kernel = row["explainer"] == "shap" and row["explainer_backend"] == "kernel"
    expected_mode = "repeated_kernel_shap" if kernel and repeats else (
        "not_requested" if kernel else "not_applicable"
    )
    if mode != expected_mode:
        errors.append(f"{experiment_id}: KernelSHAP stability artifact mode does not match row")
    if len(seeds) != repeats:
        errors.append(f"{experiment_id}: KernelSHAP stability seed count does not match row")
    if drift.shape != (repeats, int(row["n_total"])):
        errors.append(f"{experiment_id}: KernelSHAP repeat-drift shape does not match row")
    if drift.size and not np.isfinite(drift).all():
        errors.append(f"{experiment_id}: KernelSHAP repeat drift contains non-finite values")
    if drift.size:
        if not np.isclose(float(drift.mean()), float(row["shap_kernel_repeat_cosine_drift_mean"])):
            errors.append(f"{experiment_id}: KernelSHAP repeat-drift mean does not match artifact")
        if not np.isclose(float(drift.max()), float(row["shap_kernel_repeat_cosine_drift_max"])):
            errors.append(f"{experiment_id}: KernelSHAP repeat-drift max does not match artifact")


def _validate_explainer_diagnostics_artifact(
    path: Path, row: pd.Series, errors: list[str]
) -> None:
    """Check the human-readable diagnostic artifact against its result summary."""
    experiment_id = str(row["experiment_id"])
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"{experiment_id}: cannot read explainer_diagnostics.json: {exc}")
        return
    summary = payload.get("summary", {})
    if payload.get("convergence_warning_count") != int(
        row["n_explainer_convergence_warnings"]
    ):
        errors.append(f"{experiment_id}: diagnostic warning count does not match row")
    if payload.get("kernel_l1_reg") != row["shap_kernel_l1_reg"]:
        errors.append(f"{experiment_id}: diagnostic KernelSHAP l1 setting does not match row")
    for name in (
        "shap_kernel_attributions_finite",
        "shap_kernel_stability_status",
    ):
        if summary.get(name) != row[name]:
            errors.append(f"{experiment_id}: diagnostic {name} does not match row")
    for name in (
        "shap_kernel_max_additivity_error",
        "shap_kernel_repeat_cosine_drift_mean",
        "shap_kernel_repeat_cosine_drift_max",
    ):
        try:
            matches = np.isclose(float(summary[name]), float(row[name]))
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            errors.append(f"{experiment_id}: diagnostic {name} does not match row")


def _validate_noise_floor_artifact(path: Path, row: pd.Series, errors: list[str]) -> None:
    """Check the schema-3.2 repeated benign-noise evidence file."""
    experiment_id = row["experiment_id"]
    if not path.is_file():
        return  # The generic required-artifact error is clearer.
    try:
        with np.load(path, allow_pickle=False) as artifact:
            required = {
                "mode",
                "seeds",
                "benign_drift_cosine",
                "benign_drift_l1",
                "benign_drift_l2",
            }
            missing = sorted(required - set(artifact.files))
            if missing:
                errors.append(
                    f"{experiment_id}: noise_floor.npz missing arrays: {', '.join(missing)}"
                )
                return
            mode = str(artifact["mode"].item())
            seeds = np.asarray(artifact["seeds"], dtype=int)
            if mode != row["noise_floor_mode"]:
                errors.append(f"{experiment_id}: noise-floor mode does not match result row")
            if len(seeds) != int(row["noise_floor_repeats"]):
                errors.append(
                    f"{experiment_id}: noise-floor repeat count does not match result row"
                )
            if len(seeds) != len(set(seeds.tolist())):
                errors.append(f"{experiment_id}: noise-floor seeds are not unique")
            for name in ("cosine", "l1", "l2"):
                samples = np.asarray(artifact[f"benign_drift_{name}"], dtype=float)
                expected_shape = (len(seeds), int(row["n_total"]))
                if samples.shape != expected_shape:
                    errors.append(
                        f"{experiment_id}: benign {name} drift shape does not match repeats/flows"
                    )
                elif not np.isfinite(samples).all():
                    errors.append(
                        f"{experiment_id}: benign {name} drift contains non-finite values"
                    )
            if mode == "repeated_benign_lime" and len(seeds) < 2:
                errors.append(f"{experiment_id}: repeated LIME noise requires at least two seeds")
    except (OSError, ValueError) as exc:
        errors.append(f"{experiment_id}: cannot read noise_floor.npz: {exc}")
