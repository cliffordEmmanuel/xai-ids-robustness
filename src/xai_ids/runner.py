"""Benchmark runner: the full experiment pipeline and shared helpers.

``run_experiment(config)`` chains dataset -> (cached) model -> explainer -> attack
-> metrics, writes per-run artifacts, and appends one row to the master
``results.csv``. Models are cached by their ``(dataset, model, seed)`` identity so
they are trained once and reused across every explainer/strand/epsilon cell.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .attacks import ATTACKS  # registers pgd/fgsm
from .datasets import build_dataset  # registers datasets
from .explainers import EXPLAINERS  # registers shap/lime
from .metrics import compute_metrics
from .metrics.drift import cosine_drift_per_flow, l1_drift_per_flow, l2_drift_per_flow
from .models import MODELS, assess_model_quality, evaluate_classification  # registers xgboost/mlp
from .oracles import ExplanationOracle, ProbabilityOracle
from .reproducibility import make_run_dir, set_seed, setup_logging, write_provenance
from .results import RESULTS_SCHEMA_VERSION, write_results_manifest, write_run_manifest

_log = logging.getLogger("xai_ids")

DS_ABBR = {"cicids2017": "cicids", "unsw_nb15": "unsw"}
STRAND_ABBR = {"fragility": "frag", "tamper": "tamp"}


def results_dir(cfg) -> Path:
    """Return the isolated output directory for this results schema/version."""
    return Path(cfg.run.output_dir) / cfg.run.results_version


def threat_model_record(cfg) -> dict:
    """Return the declared attack-service boundary in a serialisable form."""
    budget = cfg.threat_model.query_budget
    return {
        "model_access": cfg.threat_model.model_access,
        "explanation_access": cfg.threat_model.explanation_access,
        "query_budget": {
            "model_queries": budget.model_queries,
            "explanation_queries": budget.explanation_queries,
        },
        "accounting": {
            "model_query_unit": "individual input probability vector",
            "explanation_query_unit": "logical explanation-service request per batch",
            "internal_explainer_model_queries": "not_observed_at_oracle_boundary",
        },
    }


def provenance_extra(bundle, *, cfg=None, include_fidelity: bool = False) -> dict:
    """Return serialisable preprocessing, threat-model, and fidelity provenance."""
    extra = {"preprocessing": bundle.meta}
    if cfg is not None:
        extra["threat_model"] = threat_model_record(cfg)
    if include_fidelity:
        extra["fidelity"] = {
            "reference_baseline": bundle.fidelity_baseline.tolist(),
            "feature_groups": bundle.feature_groups,
            "source": bundle.meta["fidelity_reference"]["source"],
        }
    return extra


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #
def train_slug(cfg) -> str:
    return f"{DS_ABBR.get(cfg.dataset.name, cfg.dataset.name)}-{cfg.model.name}-s{cfg.run.seed}"


def explain_slug(cfg) -> str:
    ds = DS_ABBR.get(cfg.dataset.name, cfg.dataset.name)
    variant = _explainer_variant_suffix(cfg)
    return f"{ds}-{cfg.model.name}-{cfg.explainer.name}{variant}-s{cfg.run.seed}"


def _explainer_variant_suffix(cfg) -> str:
    """Return an ID suffix whenever a noncanonical explainer variant is used."""
    if cfg.explainer.name != "shap":
        return ""
    canonical_backend = "tree" if cfg.model.name == "xgboost" else "kernel"
    backend = (
        f"-s{cfg.explainer.shap_backend}"
        if cfg.explainer.shap_backend != canonical_backend
        else ""
    )
    kernel_samples = (
        f"-ks{cfg.explainer.shap_kernel_nsamples}"
        if cfg.explainer.shap_kernel_nsamples != 100
        else ""
    )
    kernel_l1 = (
        f"-kl1{_kernel_l1_reg_label(cfg.explainer.shap_kernel_l1_reg)}"
        if cfg.explainer.shap_backend == "kernel"
        else ""
    )
    background = (
        f"-kb{cfg.explainer.shap_background_size}"
        if cfg.explainer.shap_background_size != 100
        else ""
    )
    return f"{backend}{kernel_samples}{kernel_l1}{background}"


def _kernel_l1_reg_label(value: float | str) -> str:
    """Return a filesystem-safe, stable label for an explicit KernelSHAP setting."""
    if isinstance(value, str):
        return value.replace("num_features(", "nf").replace(")", "")
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def experiment_slug(cfg) -> str:
    ds = DS_ABBR.get(cfg.dataset.name, cfg.dataset.name)
    strand = STRAND_ABBR.get(cfg.attack.strand, cfg.attack.strand)
    # LIME's sample count changes the stochastic-explainer implementation. Keep
    # the historical/default 1,500 slug stable, but disambiguate ablation cells.
    sample_suffix = (
        f"-ls{cfg.explainer.lime_num_samples}"
        if cfg.explainer.name == "lime" and cfg.explainer.lime_num_samples != 1500
        else ""
    )
    explainer_variant_suffix = _explainer_variant_suffix(cfg)
    # Suppression is the primary design. A rank-margin sensitivity run changes
    # the optimisation problem, so it must never overwrite that primary cell.
    target_objective_suffix = (
        "-rm"
        if cfg.attack.strand == "tamper" and cfg.attack.target_objective == "rank_margin"
        else ""
    )
    return (
        f"{ds}-{cfg.model.name}-{cfg.explainer.name}-{strand}"
        f"{sample_suffix}{explainer_variant_suffix}{target_objective_suffix}-{cfg.attack.method}"
        f"-e{cfg.attack.epsilon}-s{cfg.run.seed}"
    )


# --------------------------------------------------------------------------- #
# Component factories
# --------------------------------------------------------------------------- #
def make_explainer(
    cfg, *, seed: int | None = None, kernel_random_seed: int | None = None
):
    """Build an explainer, optionally as a fresh stochastic realisation."""
    seed = cfg.run.seed if seed is None else seed
    if cfg.explainer.name == "shap":
        return EXPLAINERS.get("shap")(
            backend=cfg.explainer.shap_backend,
            background_size=cfg.explainer.shap_background_size,
            kernel_nsamples=cfg.explainer.shap_kernel_nsamples,
            kernel_l1_reg=cfg.explainer.shap_kernel_l1_reg,
            seed=seed,
            kernel_random_seed=kernel_random_seed,
        )
    return EXPLAINERS.get("lime")(num_samples=cfg.explainer.lime_num_samples, seed=seed)


_PER_FLOW_DRIFT = {
    "cosine": cosine_drift_per_flow,
    "l1": l1_drift_per_flow,
    "l2": l2_drift_per_flow,
}


def collect_benign_lime_noise(cfg, model, X, baseline, background) -> dict:
    """Collect fresh-seed benign LIME drift samples for every attacked flow.

    The main baseline explanation uses the run seed. Each repeat below creates a
    new LIME object from an explicit, distinct seed; it is never a repeated call
    on the in-loop explainer. Other explainers write an empty, marked artifact so
    every schema-3.2 run has the same auditable artifact layout.
    """
    X = np.asarray(X, dtype=float)
    empty = {name: np.empty((0, len(X)), dtype=float) for name in _PER_FLOW_DRIFT}
    if cfg.explainer.name != "lime":
        return {"mode": "not_applicable", "seeds": np.array([], dtype=int), "drift": empty}
    if not cfg.metrics.measure_lime_noise_floor:
        return {"mode": "disabled", "seeds": np.array([], dtype=int), "drift": empty}

    seeds = np.asarray(
        cfg.metrics.noise_floor_seeds[: cfg.metrics.noise_floor_repeats], dtype=int
    )
    samples = {name: [] for name in _PER_FLOW_DRIFT}
    for seed in seeds:
        benign_explainer = make_explainer(cfg, seed=int(seed))
        benign = benign_explainer.explain(model, X, background=background)
        for name, drift_fn in _PER_FLOW_DRIFT.items():
            samples[name].append(drift_fn(baseline, benign))
    return {
        "mode": "repeated_benign_lime",
        "seeds": seeds,
        "drift": {name: np.stack(values) for name, values in samples.items()},
    }


def write_noise_floor_artifact(path: Path, noise: dict) -> None:
    """Persist per-flow benign drift samples in a compact, self-describing NPZ."""
    payload = {
        "mode": np.asarray(noise["mode"]),
        "seeds": np.asarray(noise["seeds"], dtype=int),
        **{f"benign_drift_{name}": values for name, values in noise["drift"].items()},
    }
    np.savez_compressed(path, **payload)


def _is_kernel_shap(cfg) -> bool:
    return cfg.explainer.name == "shap" and cfg.explainer.shap_backend == "kernel"


def _explainer_numerical_diagnostics(explainer) -> dict[str, float | int | bool]:
    """Retrieve optional explainer diagnostics without coupling runner to SHAP."""
    method = getattr(explainer, "numerical_diagnostics", None)
    if callable(method):
        return method()
    return {
        "calls": 0,
        "attributions_finite": True,
        "max_additivity_error": 0.0,
        "mean_additivity_error": 0.0,
    }


def collect_kernel_shap_stability(cfg, model, X, baseline, background) -> dict:
    """Measure fresh-seed KernelSHAP reproducibility on the unperturbed flows.

    These are evaluator-side calls. They are deliberately independent from the
    attack explanation-oracle budget and use the same run-seeded background
    sample while varying only KernelSHAP's mask-sampling seed.
    """
    X = np.asarray(X, dtype=float)
    empty = np.empty((0, len(X)), dtype=float)
    if not _is_kernel_shap(cfg):
        return {
            "mode": "not_applicable",
            "seeds": np.array([], dtype=int),
            "cosine_drift": empty,
            "repeat_diagnostics": [],
        }
    repeats = cfg.explainer.shap_kernel_stability_repeats
    if not repeats:
        return {
            "mode": "not_requested",
            "seeds": np.array([], dtype=int),
            "cosine_drift": empty,
            "repeat_diagnostics": [],
        }

    seeds = np.asarray(cfg.explainer.shap_kernel_stability_seeds[:repeats], dtype=int)
    drift_samples: list[np.ndarray] = []
    repeat_diagnostics: list[dict[str, float | int | bool]] = []
    for random_seed in seeds:
        repeat_explainer = make_explainer(
            cfg,
            seed=cfg.run.seed,
            kernel_random_seed=int(random_seed),
        )
        repeat = repeat_explainer.explain(model, X, background=background)
        drift_samples.append(cosine_drift_per_flow(baseline, repeat))
        repeat_diagnostics.append(_explainer_numerical_diagnostics(repeat_explainer))
    return {
        "mode": "repeated_kernel_shap",
        "seeds": seeds,
        "cosine_drift": np.stack(drift_samples),
        "repeat_diagnostics": repeat_diagnostics,
    }


def write_kernel_shap_stability_artifact(path: Path, stability: dict) -> None:
    """Persist per-flow KernelSHAP repeat drift, including a marked empty mode."""
    np.savez_compressed(
        path,
        mode=np.asarray(stability["mode"]),
        seeds=np.asarray(stability["seeds"], dtype=int),
        repeat_cosine_drift=np.asarray(stability["cosine_drift"], dtype=float),
    )


def summarise_kernel_shap_stability(
    cfg,
    *,
    main_diagnostics: dict[str, float | int | bool],
    repeat_stability: dict,
    convergence_warning_count: int,
) -> dict[str, float | int | bool | str]:
    """Summarise the predeclared numerical-stability rule for a result row."""
    if not _is_kernel_shap(cfg):
        return {
            "shap_kernel_attributions_finite": True,
            "shap_kernel_max_additivity_error": 0.0,
            "shap_kernel_repeat_cosine_drift_mean": 0.0,
            "shap_kernel_repeat_cosine_drift_max": 0.0,
            "shap_kernel_stability_status": "not_applicable",
        }

    diagnostics = [main_diagnostics, *repeat_stability["repeat_diagnostics"]]
    finite = all(bool(record["attributions_finite"]) for record in diagnostics)
    additivity_errors = [float(record["max_additivity_error"]) for record in diagnostics]
    max_additivity_error = max(additivity_errors) if additivity_errors else 0.0
    drift = np.asarray(repeat_stability["cosine_drift"], dtype=float)
    mean_drift = float(np.mean(drift)) if drift.size else 0.0
    max_drift = float(np.max(drift)) if drift.size else 0.0
    repeats_complete = len(repeat_stability["seeds"]) >= 2
    checks_pass = (
        convergence_warning_count == 0
        and finite
        and np.isfinite(max_additivity_error)
        and max_additivity_error <= cfg.explainer.shap_kernel_additivity_tolerance
        and repeats_complete
        and np.isfinite(max_drift)
        and max_drift <= cfg.explainer.shap_kernel_repeat_cosine_drift_tolerance
    )
    if not finite or convergence_warning_count or not np.isfinite(max_additivity_error):
        status = "fail"
    elif not repeats_complete:
        status = "not_checked"
    else:
        status = "pass" if checks_pass else "fail"
    return {
        "shap_kernel_attributions_finite": finite,
        "shap_kernel_max_additivity_error": float(max_additivity_error),
        "shap_kernel_repeat_cosine_drift_mean": mean_drift,
        "shap_kernel_repeat_cosine_drift_max": max_drift,
        "shap_kernel_stability_status": status,
    }


def make_attack(cfg):
    a = cfg.attack
    # Canonical FGSM = a single full-epsilon step.
    step_size = a.epsilon if a.method == "fgsm" else a.step_size
    return ATTACKS.get(a.method)(
        epsilon=a.epsilon,
        steps=a.steps,
        step_size=step_size,
        spsa_q=a.spsa.q,
        spsa_delta=a.spsa.delta,
        target_rule=a.target_rule,
        target_objective=a.target_objective,
        target_topk=cfg.metrics.topk,
        enforce_prediction_preserved=a.enforce_prediction_preserved,
        seed=cfg.run.seed,
    )


def make_attack_oracles(cfg, model, explainer, background):
    """Bind runner-owned components into the attack's restricted service APIs."""
    budget = cfg.threat_model.query_budget
    model_oracle = ProbabilityOracle.from_model(model, budget=budget.model_queries)
    explanation_oracle = None
    if cfg.attack.strand == "tamper":
        if cfg.threat_model.explanation_access != "oracle":
            raise ValueError("tamper attacks require an explanation oracle")
        explanation_oracle = ExplanationOracle.from_components(
            explainer,
            model,
            background=background,
            budget=budget.explanation_queries,
        )
    return model_oracle, explanation_oracle


def _model_params(cfg) -> dict:
    """Model hyperparameters with the run seed threaded in (for reproducibility)."""
    params = dict(cfg.model.params)
    if cfg.model.name == "mlp":
        params.setdefault("seed", cfg.run.seed)
    elif cfg.model.name == "xgboost":
        params.setdefault("random_state", cfg.run.seed)
    return params


def _model_fingerprint(cfg, bundle) -> str:
    """Hash of the config fields that actually determine the trained model."""
    payload = json.dumps(
        {
            "model": cfg.model.name,
            "class_weight": cfg.model.class_weight,
            "params": _model_params(cfg),
            "dataset": cfg.dataset.name,
            "normalise": cfg.dataset.normalise,
            "balance": cfg.dataset.balance,
            "test_size": cfg.dataset.test_size,
            "seed": cfg.run.seed,
            "preprocessing": bundle.meta.get("preprocessing_schema_version"),
            "categorical_transformer": bundle.meta.get("categorical_transformer"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def true_positive_subset(model, bundle, size, seed):
    """Flows the model correctly flags as attacks (y==1 and predict==1)."""
    preds = model.predict(bundle.X_test)
    tp = np.where((bundle.y_test == 1) & (preds == 1))[0]
    if len(tp) > size:
        rng = np.random.default_rng(seed)
        tp = np.sort(rng.choice(tp, size, replace=False))
    return bundle.X_test[tp], tp


# --------------------------------------------------------------------------- #
# Model cache
# --------------------------------------------------------------------------- #
def load_or_train_model(cfg, bundle):
    """Load the cached model for (dataset, model, seed), or train and cache it.

    A fingerprint of the model/preprocessing-relevant config is stored alongside
    the model; a cached model is only reused if that fingerprint matches, so
    changing hyperparameters, balancing, or normalisation forces a retrain.
    """
    model_cls = MODELS.get(cfg.model.name)
    train_dir = results_dir(cfg) / train_slug(cfg)
    model_path = train_dir / f"model{model_cls.ext}"
    fp_path = train_dir / "fingerprint.txt"
    fingerprint = _model_fingerprint(cfg, bundle)

    if model_path.exists() and fp_path.exists() and fp_path.read_text().strip() == fingerprint:
        _log.info("loading cached model %s", model_path)
        model = model_cls.load(model_path)
        metrics = model_quality_record(model, bundle)
        (train_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return model

    _log.info("training %s (class_weight=%s) ...", cfg.model.name, cfg.model.class_weight)
    model = model_cls(class_weight=cfg.model.class_weight, params=_model_params(cfg))
    model.fit(bundle.X_train, bundle.y_train)
    metrics = model_quality_record(model, bundle)
    train_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(train_dir, cfg, cfg.run.seed, extra=provenance_extra(bundle, cfg=cfg))
    model.save(model_path)
    fp_path.write_text(fingerprint)
    (train_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _log.info("model metrics: %s", metrics)
    return model


def model_quality_record(model, bundle) -> dict[str, float | int | bool]:
    """Evaluate the frozen test set and attach the declared comparison policy."""
    return assess_model_quality(evaluate_classification(model, bundle.X_test, bundle.y_test))


# --------------------------------------------------------------------------- #
# Results table
# --------------------------------------------------------------------------- #
def _identity(cfg) -> dict:
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "results_version": cfg.run.results_version,
        "experiment_id": experiment_slug(cfg),
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "explainer": cfg.explainer.name,
        "explainer_backend": (
            cfg.explainer.shap_backend if cfg.explainer.name == "shap" else "lime"
        ),
        "shap_background_size": cfg.explainer.shap_background_size,
        "shap_kernel_nsamples": cfg.explainer.shap_kernel_nsamples,
        "shap_kernel_l1_reg": (
            _kernel_l1_reg_label(cfg.explainer.shap_kernel_l1_reg)
            if _is_kernel_shap(cfg)
            else "not_applicable"
        ),
        "shap_kernel_stability_repeats": (
            cfg.explainer.shap_kernel_stability_repeats if _is_kernel_shap(cfg) else 0
        ),
        "shap_kernel_additivity_tolerance": (
            cfg.explainer.shap_kernel_additivity_tolerance if _is_kernel_shap(cfg) else 0.0
        ),
        "shap_kernel_repeat_cosine_drift_tolerance": (
            cfg.explainer.shap_kernel_repeat_cosine_drift_tolerance
            if _is_kernel_shap(cfg)
            else 0.0
        ),
        "require_shap_kernel_stability": (
            cfg.explainer.require_shap_kernel_stability if _is_kernel_shap(cfg) else False
        ),
        "lime_num_samples": cfg.explainer.lime_num_samples,
        "strand": cfg.attack.strand,
        "method": cfg.attack.method,
        "objective": cfg.attack.objective,
        "target_rule": cfg.attack.target_rule,
        "target_objective": cfg.attack.target_objective,
        "target_topk": cfg.metrics.topk,
        "gradient_source": cfg.attack.gradient_source,
        "threat_model_model_access": cfg.threat_model.model_access,
        "threat_model_explanation_access": cfg.threat_model.explanation_access,
        "model_query_budget": cfg.threat_model.query_budget.model_queries,
        "explanation_query_budget": cfg.threat_model.query_budget.explanation_queries,
        "noise_floor_mode": (
            "repeated_benign_lime"
            if cfg.explainer.name == "lime" and cfg.metrics.measure_lime_noise_floor
            else "disabled" if cfg.explainer.name == "lime" else "not_applicable"
        ),
        "noise_floor_repeats": (
            cfg.metrics.noise_floor_repeats
            if cfg.explainer.name == "lime" and cfg.metrics.measure_lime_noise_floor
            else 0
        ),
        "epsilon": cfg.attack.epsilon,
        "seed": cfg.run.seed,
    }


def append_results_row(output_dir, row: dict) -> Path:
    """Append (idempotently, keyed by experiment_id) one row to results.csv."""
    path = Path(output_dir) / "results.csv"
    new = pd.DataFrame([row])
    if path.exists():
        df = pd.read_csv(path)
        if "schema_version" in df:
            existing_schemas = set(df["schema_version"].dropna().astype(str))
            if existing_schemas and existing_schemas != {RESULTS_SCHEMA_VERSION}:
                raise ValueError(
                    f"cannot append schema {RESULTS_SCHEMA_VERSION} to {path} "
                    f"with existing schema(s) {sorted(existing_schemas)}; "
                    "start a new results_version instead"
                )
        df = df[df["experiment_id"] != row["experiment_id"]]
        df = pd.concat([df, new], ignore_index=True)
    else:
        df = new
    df.to_csv(path, index=False)
    write_results_manifest(Path(output_dir), df)
    return path


def build_per_flow_frame(
    cfg, *, tp_indices, X_base, result, adversarial_attributions: np.ndarray | None = None
) -> pd.DataFrame:
    """Return row-level attack evidence aligned with the experiment summary.

    This deliberately stores facts about each attacked flow, rather than an
    opaque success flag: final decisions, the in-loop freeze safeguard, exact
    perturbation magnitudes, and (for tamper) target attribution/rank outcomes.
    The target baseline is the exact explanation realization used inside the
    attack to select that target; all supporting outcome fields are traceable to
    checked artifacts without re-running the attack.
    """
    X_base = np.asarray(X_base, dtype=float)
    X_adv = np.asarray(result.X_adv, dtype=float)
    n = len(X_base)
    if X_adv.shape != X_base.shape:
        raise ValueError("X_base and result.X_adv must have matching shapes.")

    reference_predictions = np.asarray(result.meta["reference_predictions"])
    adversarial_predictions = np.asarray(result.meta["adversarial_predictions"])
    frozen = np.asarray(
        result.meta.get("frozen_due_prediction_flip", np.zeros(n, dtype=bool)), dtype=bool
    )
    preserved = np.asarray(result.prediction_preserved, dtype=bool)
    evidence = (reference_predictions, adversarial_predictions, frozen, preserved)
    if not all(len(values) == n for values in evidence):
        raise ValueError("Attack metadata must align with every attacked flow.")
    if not np.array_equal(preserved, reference_predictions == adversarial_predictions):
        raise ValueError("Attack preservation mask must agree with stored predictions.")

    targets = result.meta.get("targets")
    target_feature = np.full(n, np.nan)
    target_applicable = np.zeros(n, dtype=bool)
    target_baseline_abs = np.full(n, np.nan)
    target_adversarial_abs = np.full(n, np.nan)
    target_relative_reduction = np.full(n, np.nan)
    target_baseline_rank = np.full(n, np.nan)
    target_adversarial_rank = np.full(n, np.nan)
    target_rank_deterioration = np.full(n, np.nan)
    target_in_topk_baseline = np.zeros(n, dtype=bool)
    target_in_topk_adversarial = np.zeros(n, dtype=bool)
    target_success = np.zeros(n, dtype=bool)
    target_topk = np.full(n, np.nan)
    target_objective = np.full(n, "not_applicable", dtype=object)
    if targets is not None:
        if len(targets) != n:
            raise ValueError("Attack targets must align with every attacked flow.")
        target_feature = np.asarray(targets, dtype=float)
        objective_baseline = result.meta.get("objective_baseline_attributions")
        if objective_baseline is None or adversarial_attributions is None:
            raise ValueError(
                "Tamper outcomes require objective baseline and adversarial attributions."
            )
        objective_baseline = np.asarray(objective_baseline, dtype=float)
        adversarial_attributions = np.asarray(adversarial_attributions, dtype=float)
        if (
            objective_baseline.shape != adversarial_attributions.shape
            or objective_baseline.shape[0] != n
        ):
            raise ValueError("Tamper attribution evidence must align with every attacked flow.")
        target_idx = np.asarray(targets, dtype=int)
        if (
            target_idx.ndim != 1
            or (target_idx < 0).any()
            or (target_idx >= objective_baseline.shape[1]).any()
        ):
            raise ValueError("Tamper targets must be valid feature indices.")
        rows = np.arange(n)
        target_applicable[:] = True
        target_topk[:] = cfg.metrics.topk
        target_objective[:] = result.meta["target_objective"]
        target_baseline_abs = np.abs(objective_baseline[rows, target_idx])
        target_adversarial_abs = np.abs(adversarial_attributions[rows, target_idx])
        target_relative_reduction = np.divide(
            target_baseline_abs - target_adversarial_abs,
            target_baseline_abs,
            out=np.full(n, np.nan),
            where=target_baseline_abs > 0,
        )
        # Stable ordering resolves equal-magnitude ties by feature index. The
        # same deterministic policy is used by the result validator.
        baseline_order = np.argsort(-np.abs(objective_baseline), axis=1, kind="stable")
        adversarial_order = np.argsort(
            -np.abs(adversarial_attributions), axis=1, kind="stable"
        )
        target_baseline_rank = (baseline_order == target_idx[:, None]).argmax(axis=1) + 1
        target_adversarial_rank = (adversarial_order == target_idx[:, None]).argmax(axis=1) + 1
        target_rank_deterioration = target_adversarial_rank - target_baseline_rank
        target_in_topk_baseline = target_baseline_rank <= cfg.metrics.topk
        target_in_topk_adversarial = target_adversarial_rank <= cfg.metrics.topk
        target_success = target_adversarial_rank > cfg.metrics.topk

    delta = X_adv - X_base
    changed = np.count_nonzero(~np.isclose(delta, 0.0, atol=1e-12), axis=1)
    return pd.DataFrame(
        {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "results_version": cfg.run.results_version,
            "experiment_id": experiment_slug(cfg),
            "flow_index": np.arange(n, dtype=int),
            "test_index": np.asarray(tp_indices, dtype=int),
            "reference_prediction": reference_predictions,
            "adversarial_prediction": adversarial_predictions,
            "prediction_preserved": preserved,
            "frozen_due_prediction_flip": frozen,
            "perturbation_linf": np.max(np.abs(delta), axis=1),
            "perturbation_l1": np.abs(delta).sum(axis=1),
            "perturbation_l2": np.linalg.norm(delta, axis=1),
            "n_changed_features": changed,
            "target_feature": target_feature,
            "target_applicable": target_applicable,
            "target_objective": target_objective,
            "target_topk": target_topk,
            "target_baseline_abs_attribution": target_baseline_abs,
            "target_adversarial_abs_attribution": target_adversarial_abs,
            "target_relative_attribution_reduction": target_relative_reduction,
            "target_baseline_rank": target_baseline_rank,
            "target_adversarial_rank": target_adversarial_rank,
            "target_rank_deterioration": target_rank_deterioration,
            "target_in_topk_baseline": target_in_topk_baseline,
            "target_in_topk_adversarial": target_in_topk_adversarial,
            "target_success_topk_displacement": target_success,
        }
    )


def tamper_outcomes_from_frame(per_flow: pd.DataFrame) -> dict[str, np.ndarray] | None:
    """Return exact target-outcome arrays for cohort-specific aggregation."""
    applicable = per_flow["target_applicable"].to_numpy(dtype=bool)
    if not applicable.any():
        return None
    if not applicable.all():
        raise ValueError("A tamper run must define target evidence for every attacked flow.")
    return {
        "success_topk_displacement": per_flow[
            "target_success_topk_displacement"
        ].to_numpy(dtype=bool),
        "baseline_abs_attribution": per_flow["target_baseline_abs_attribution"].to_numpy(),
        "adversarial_abs_attribution": per_flow[
            "target_adversarial_abs_attribution"
        ].to_numpy(),
        "relative_attribution_reduction": per_flow[
            "target_relative_attribution_reduction"
        ].to_numpy(),
        "baseline_rank": per_flow["target_baseline_rank"].to_numpy(),
        "adversarial_rank": per_flow["target_adversarial_rank"].to_numpy(),
        "rank_deterioration": per_flow["target_rank_deterioration"].to_numpy(),
        "in_topk_baseline": per_flow["target_in_topk_baseline"].to_numpy(dtype=bool),
        "in_topk_adversarial": per_flow[
            "target_in_topk_adversarial"
        ].to_numpy(dtype=bool),
    }


def _prefix_metrics(prefix: str, metrics: dict) -> dict:
    """Namespace cohort metrics while retaining one unambiguous retention field."""
    return {f"{prefix}{name}": value for name, value in metrics.items() if name != "retention_rate"}


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def run_experiment(cfg) -> dict:
    set_seed(cfg.run.seed)
    output_dir = results_dir(cfg)
    run_dir = make_run_dir(output_dir, experiment_slug(cfg))
    log = setup_logging(run_dir)
    log.info("run dir: %s", run_dir)

    log.info("loading + preprocessing dataset '%s' ...", cfg.dataset.name)
    bundle = build_dataset(cfg)
    write_provenance(
        run_dir,
        cfg,
        cfg.run.seed,
        extra=provenance_extra(bundle, cfg=cfg, include_fidelity=True),
    )

    model = load_or_train_model(cfg, bundle)
    model_quality = model_quality_record(model, bundle)

    X_tp, tp_idx = true_positive_subset(
        model, bundle, cfg.dataset.attack_subset_size, cfg.run.seed
    )
    log.info("true-positive attack subset: %d flows", len(X_tp))

    explainer = make_explainer(cfg)
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        log.info("baseline %s attributions ...", cfg.explainer.name)
        baseline = explainer.explain(model, X_tp, background=bundle.X_train)

        log.info(
            "attack: %s strand=%s objective=%s eps=%s steps=%d",
            cfg.attack.method,
            cfg.attack.strand,
            cfg.attack.objective,
            cfg.attack.epsilon,
            cfg.attack.steps,
        )
        model_oracle, explanation_oracle = make_attack_oracles(
            cfg, model, explainer, background=bundle.X_train
        )
        result = make_attack(cfg).generate(
            model_oracle,
            X_tp,
            objective=cfg.attack.objective,
            gradient_source=cfg.attack.gradient_source,
            explanation_oracle=explanation_oracle,
        )
        adversarial = explainer.explain(model, result.X_adv)
        noise = collect_benign_lime_noise(cfg, model, X_tp, baseline, bundle.X_train)
        kernel_stability = collect_kernel_shap_stability(
            cfg, model, X_tp, baseline, bundle.X_train
        )
    convergence_messages = sorted(
        {
            str(warning.message)
            for warning in captured_warnings
            if issubclass(warning.category, ConvergenceWarning)
        }
    )
    n_explainer_convergence_warnings = sum(
        issubclass(warning.category, ConvergenceWarning) for warning in captured_warnings
    )
    main_explainer_diagnostics = _explainer_numerical_diagnostics(explainer)
    kernel_stability_summary = summarise_kernel_shap_stability(
        cfg,
        main_diagnostics=main_explainer_diagnostics,
        repeat_stability=kernel_stability,
        convergence_warning_count=n_explainer_convergence_warnings,
    )
    write_noise_floor_artifact(run_dir / "noise_floor.npz", noise)
    write_kernel_shap_stability_artifact(run_dir / "kernel_shap_stability.npz", kernel_stability)

    np.savez_compressed(
        run_dir / "attack.npz",
        X_adv=result.X_adv,
        tp_indices=tp_idx,
        baseline_attributions=baseline,
        adversarial_attributions=adversarial,
        prediction_preserved=result.prediction_preserved,
        reference_predictions=np.asarray(result.meta["reference_predictions"]),
        adversarial_predictions=np.asarray(result.meta["adversarial_predictions"]),
        frozen_due_prediction_flip=np.asarray(result.meta["frozen_due_prediction_flip"]),
        targets=np.asarray(
            [] if result.meta["targets"] is None else result.meta["targets"], dtype=int
        ),
        objective_baseline_attributions=np.asarray(
            np.empty((0, baseline.shape[1]))
            if result.meta["objective_baseline_attributions"] is None
            else result.meta["objective_baseline_attributions"]
        ),
        target_objective=np.asarray(result.meta["target_objective"]),
        target_topk=np.asarray(result.meta["target_topk"], dtype=int),
    )
    per_flow = build_per_flow_frame(
        cfg,
        tp_indices=tp_idx,
        X_base=X_tp,
        result=result,
        adversarial_attributions=adversarial,
    )
    per_flow.to_csv(run_dir / "per_flow.csv", index=False)
    tamper_outcomes = tamper_outcomes_from_frame(per_flow)

    log.info("computing metrics ...")
    all_metrics = compute_metrics(
        baseline,
        adversarial,
        X_base=X_tp,
        X_adv=result.X_adv,
        model=model,
        preserved=result.prediction_preserved,
        metrics_cfg=cfg.metrics,
        tamper_outcomes=tamper_outcomes,
        fidelity_baseline=bundle.fidelity_baseline,
        feature_groups=bundle.feature_groups,
        benign_drift_samples=noise["drift"] if noise["mode"] == "repeated_benign_lime" else None,
        require_prediction_preserved=False,
    )
    preserved_metrics = compute_metrics(
        baseline,
        adversarial,
        X_base=X_tp,
        X_adv=result.X_adv,
        model=model,
        preserved=result.prediction_preserved,
        metrics_cfg=cfg.metrics,
        tamper_outcomes=tamper_outcomes,
        fidelity_baseline=bundle.fidelity_baseline,
        feature_groups=bundle.feature_groups,
        benign_drift_samples=noise["drift"] if noise["mode"] == "repeated_benign_lime" else None,
        require_prediction_preserved=True,
    )
    n_total = len(result.prediction_preserved)
    n_preserved = int(result.prediction_preserved.sum())
    summary = {
        "n_total": n_total,
        "n_preserved": n_preserved,
        "n_flipped": n_total - n_preserved,
        "retention_rate": result.retention_rate,
        "n_frozen_due_prediction_flip": result.meta["n_frozen_due_prediction_flip"],
        "model_queries": result.meta["model_queries"],
        "explanation_queries": result.meta["explanation_queries"],
        "n_perturbed": result.meta["n_perturbed"],
        "n_explainer_convergence_warnings": n_explainer_convergence_warnings,
        **kernel_stability_summary,
    }
    metrics = {
        **summary,
        **_prefix_metrics("all_", all_metrics),
        **_prefix_metrics("preserved_", preserved_metrics),
    }
    metrics_payload = {
        "summary": summary,
        "model_quality": model_quality,
        "attack_oracle_accounting": {
            "model": result.meta["model_oracle"],
            "explanation": result.meta["explanation_oracle"],
        },
        "noise_floor": {
            "mode": noise["mode"],
            "seeds": noise["seeds"].tolist(),
            "per_flow_artifact": "noise_floor.npz",
        },
        "explainer_warnings": {
            "convergence_warning_count": n_explainer_convergence_warnings,
            "convergence_warning_messages": convergence_messages,
        },
        "kernel_shap_numerical_stability": {
            "summary": kernel_stability_summary,
            "main_explainer_diagnostics": main_explainer_diagnostics,
            "repeat_mode": kernel_stability["mode"],
            "repeat_seeds": kernel_stability["seeds"].tolist(),
            "repeat_diagnostics": kernel_stability["repeat_diagnostics"],
            "per_flow_artifact": "kernel_shap_stability.npz",
        },
        "all_attacked": all_metrics,
        "prediction_preserved": preserved_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (run_dir / "fidelity.json").write_text(
        json.dumps(
            {
                "source": bundle.meta["fidelity_reference"]["source"],
                "reference_baseline": bundle.fidelity_baseline.tolist(),
                "feature_groups": bundle.feature_groups,
            },
            indent=2,
        )
    )
    (run_dir / "explainer_diagnostics.json").write_text(
        json.dumps(
            {
                "kind": "explanation-numerical-stability",
                "explainer": cfg.explainer.name,
                "backend": (
                    cfg.explainer.shap_backend if cfg.explainer.name == "shap" else "lime"
                ),
                "kernel_l1_reg": _identity(cfg)["shap_kernel_l1_reg"],
                "summary": kernel_stability_summary,
                "main_explainer_diagnostics": main_explainer_diagnostics,
                "repeat_mode": kernel_stability["mode"],
                "repeat_seeds": kernel_stability["seeds"].tolist(),
                "repeat_diagnostics": kernel_stability["repeat_diagnostics"],
                "convergence_warning_count": n_explainer_convergence_warnings,
                "convergence_warning_messages": convergence_messages,
            },
            indent=2,
        )
    )

    model_quality_row = {f"model_{name}": value for name, value in model_quality.items()}
    row = {**_identity(cfg), **model_quality_row, **metrics}
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
    results_path = append_results_row(output_dir, row)
    log.info("row -> %s | retention=%.3f", results_path, summary["retention_rate"])
    return row


def run_sweep(cfgs: list) -> list[dict]:
    """Run several experiments (already-loaded configs) into the shared results.csv."""
    rows = []
    for i, cfg in enumerate(cfgs, 1):
        _log.info("[sweep %d/%d] %s", i, len(cfgs), experiment_slug(cfg))
        rows.append(run_experiment(cfg))
    return rows
