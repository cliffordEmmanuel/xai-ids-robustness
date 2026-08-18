"""Experiment configuration models and loader.

This module is the single source of truth for the YAML schema described in
``docs/implementation_workplan.md`` (section 2). Configs are parsed and validated
with pydantic v2; cross-field rules encode the Path B design decisions so that an
inconsistent experiment fails fast at load time rather than deep inside a run.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Leaf sections
# --------------------------------------------------------------------------- #


class _Base(BaseModel):
    # Reject unknown keys so typos in YAML are caught immediately.
    model_config = ConfigDict(extra="forbid")


class RunConfig(_Base):
    id: str | None = None
    seed: int = 42
    seeds: list[int] = Field(default_factory=lambda: [42, 7, 123])
    # ``output_dir`` is the stable root. New experiment results are placed in a
    # versioned subdirectory so a rerun cannot overwrite a prior evidence record.
    output_dir: str = "results"
    results_version: str = Field("v4", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DatasetConfig(_Base):
    name: Literal["cicids2017", "unsw_nb15"]
    data_dir: str | None = None  # defaults to the loader's own data/raw/<name>
    normalise: Literal["standard", "minmax", "none"] = "standard"
    # Default: handle imbalance at the model level (class weights), not by
    # resampling. undersample/smote remain available for sensitivity checks.
    balance: Literal["none", "undersample_train_only", "smote_train_only"] = "none"
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    attack_subset_size: int = Field(500, gt=0)


class ModelConfig(_Base):
    name: Literal["xgboost", "mlp"]
    class_weight: bool = True  # handle imbalance at the model level
    params: dict[str, Any] = Field(default_factory=dict)  # per-model hyperparameters


class ExplainerConfig(_Base):
    name: Literal["shap", "lime"]
    shap_backend: Literal["tree", "kernel"] = "tree"
    shap_background_size: int = Field(100, gt=0)
    shap_kernel_nsamples: int = Field(100, gt=0)
    # A positive numeric value uses SHAP's fixed-coefficient Lasso path.  The
    # legacy ``num_features(N)`` option uses LARS feature selection and is
    # retained only as an explicit calibration candidate, never an implicit
    # backend default.
    shap_kernel_l1_reg: float | str = 0.0001
    shap_kernel_stability_repeats: int = Field(0, ge=0)
    shap_kernel_stability_seeds: list[int] = Field(
        default_factory=lambda: [2003, 2011, 2017, 2027, 2029]
    )
    shap_kernel_additivity_tolerance: float = Field(1e-6, gt=0)
    shap_kernel_repeat_cosine_drift_tolerance: float = Field(0.05, ge=0)
    require_shap_kernel_stability: bool = False
    lime_num_samples: int = Field(1500, gt=0)
    cache_baseline: bool = True

    @field_validator("shap_kernel_l1_reg")
    @classmethod
    def _validate_kernel_l1_reg(cls, value: float | str) -> float | str:
        """Permit only explicit, auditable KernelSHAP regularisation modes."""
        if isinstance(value, bool):
            raise ValueError("shap_kernel_l1_reg must be a positive number or num_features(N)")
        if isinstance(value, (int, float)):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                raise ValueError("shap_kernel_l1_reg numeric values must be finite and positive")
            return numeric
        if isinstance(value, str) and re.fullmatch(r"num_features\([1-9]\d*\)", value):
            return value
        raise ValueError(
            "shap_kernel_l1_reg must be a positive number or num_features(N); "
            "implicit/automatic regularisation is not allowed"
        )


class QueryBudgetConfig(_Base):
    """Caps on attack-oracle use; units are documented in the oracle records."""

    model_queries: int = Field(125_000, gt=0)
    explanation_queries: int = Field(250, gt=0)


class ThreatModelConfig(_Base):
    """The externally visible services available to the adversary."""

    model_access: Literal["probability_query"] = "probability_query"
    explanation_access: Literal["oracle", "none"] = "oracle"
    query_budget: QueryBudgetConfig = Field(default_factory=QueryBudgetConfig)


class SpsaConfig(_Base):
    q: int = Field(2, ge=1)
    delta: float = Field(0.01, gt=0)


class AttackConfig(_Base):
    strand: Literal["fragility", "tamper"]
    method: Literal["fgsm", "pgd"]
    gradient_source: Literal["spsa", "autograd"] = "spsa"
    objective: Literal["loss", "attribution"] = "loss"
    epsilon: float = Field(0.03, gt=0)
    steps: int = Field(50, ge=1)
    step_size: float = Field(0.005, gt=0)
    spsa: SpsaConfig = Field(default_factory=SpsaConfig)
    target_rule: Literal["top1_suppress", "none"] = "top1_suppress"
    target_objective: Literal["suppression", "rank_margin"] = "suppression"
    enforce_prediction_preserved: bool = True


class MetricsConfig(_Base):
    stability: list[Literal["spearman", "topk_jaccard"]] = Field(
        default_factory=lambda: ["spearman", "topk_jaccard"]
    )
    fidelity: list[Literal["local_fidelity", "deletion", "insertion"]] = Field(
        default_factory=lambda: ["deletion", "insertion"]  # local_fidelity deferred
    )
    drift: list[Literal["cosine", "l1", "l2"]] = Field(
        default_factory=lambda: ["cosine", "l1", "l2"]
    )
    topk: int = Field(10, gt=0)
    measure_lime_noise_floor: bool = True
    noise_floor_repeats: int = Field(10, ge=2)
    noise_floor_seeds: list[int] = Field(
        default_factory=lambda: [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061]
    )


# --------------------------------------------------------------------------- #
# Root config + cross-field validation
# --------------------------------------------------------------------------- #


class ExperimentConfig(_Base):
    run: RunConfig = Field(default_factory=RunConfig)
    dataset: DatasetConfig
    model: ModelConfig
    explainer: ExplainerConfig
    threat_model: ThreatModelConfig = Field(default_factory=ThreatModelConfig)
    attack: AttackConfig
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @model_validator(mode="after")
    def _check_consistency(self) -> ExperimentConfig:
        errors: list[str] = []

        # 1. Strand <-> objective coupling (Path B).
        #    fragility uses the loss objective; tamper uses the attribution objective.
        if self.attack.strand == "tamper" and self.attack.objective != "attribution":
            errors.append(
                "attack.strand='tamper' requires attack.objective='attribution'"
            )
        if self.attack.strand == "fragility" and self.attack.objective != "loss":
            errors.append(
                "attack.strand='fragility' requires attack.objective='loss'"
            )

        # 2. autograd gradients are only defined for the differentiable MLP.
        if self.attack.gradient_source == "autograd" and self.model.name != "mlp":
            errors.append(
                "attack.gradient_source='autograd' is only valid for model.name='mlp' "
                "(XGBoost is non-differentiable; use 'spsa')"
            )

        # 3. TreeSHAP backend is only valid for tree models (XGBoost).
        if (
            self.explainer.name == "shap"
            and self.explainer.shap_backend == "tree"
            and self.model.name != "xgboost"
        ):
            errors.append(
                "explainer.shap_backend='tree' requires model.name='xgboost' "
                "(use 'kernel' for the MLP)"
            )

        # 4. The attribution objective needs an explanation to target.
        if self.attack.objective == "attribution" and self.attack.target_rule == "none":
            errors.append(
                "attack.objective='attribution' requires a target_rule other than 'none'"
            )
        if self.attack.objective == "loss" and self.attack.target_objective != "suppression":
            errors.append(
                "attack.target_objective='rank_margin' is only valid for attribution attacks"
            )

        # 5. Tamper optimisation needs an explicitly configured explanation
        # service. A model-agnostic probability oracle alone cannot supply its
        # attribution objective.
        if self.attack.strand == "tamper" and self.threat_model.explanation_access != "oracle":
            errors.append(
                "attack.strand='tamper' requires threat_model.explanation_access='oracle'"
            )

        # 6. Benign LIME replicates must have independent, explicit seeds.  The
        # stored seed list makes the stochasticity estimate reproducible.
        if len(self.metrics.noise_floor_seeds) < self.metrics.noise_floor_repeats:
            errors.append(
                "metrics.noise_floor_seeds must contain at least noise_floor_repeats values"
            )
        if len(set(self.metrics.noise_floor_seeds)) != len(self.metrics.noise_floor_seeds):
            errors.append("metrics.noise_floor_seeds must not contain duplicates")
        if (
            self.explainer.name == "lime"
            and self.run.seed in self.metrics.noise_floor_seeds[: self.metrics.noise_floor_repeats]
        ):
            errors.append(
                "metrics.noise_floor_seeds for LIME must not reuse the main run.seed"
            )

        # 7. KernelSHAP diagnostics use independently seeded, fresh explainers
        # with a fixed background sample.  Final releases may require the
        # diagnostic pass, whereas calibration runs deliberately record both
        # passing and failing candidates.
        kernel = (
            self.explainer.name == "shap" and self.explainer.shap_backend == "kernel"
        )
        repeats = self.explainer.shap_kernel_stability_repeats
        seeds = self.explainer.shap_kernel_stability_seeds
        if repeats:
            if not kernel:
                errors.append(
                    "shap_kernel_stability_repeats is only valid for KernelSHAP"
                )
            if len(seeds) < repeats:
                errors.append(
                    "explainer.shap_kernel_stability_seeds must cover "
                    "shap_kernel_stability_repeats"
                )
            if len(set(seeds)) != len(seeds):
                errors.append("explainer.shap_kernel_stability_seeds must not contain duplicates")
            if self.run.seed in seeds[:repeats]:
                errors.append(
                    "explainer.shap_kernel_stability_seeds must not reuse the main run.seed"
                )
        if self.explainer.require_shap_kernel_stability:
            if not kernel:
                errors.append(
                    "require_shap_kernel_stability is only valid for KernelSHAP"
                )
            if repeats < 2:
                errors.append(
                    "require_shap_kernel_stability requires at least two stability repeats"
                )

        if errors:
            raise ValueError("Invalid experiment configuration:\n- " + "\n- ".join(errors))

        # Soft normalisation: FGSM is a single step regardless of the configured budget.
        if self.attack.method == "fgsm":
            self.attack.steps = 1

        # Prediction preservation is determined by the strand: fragility filters
        # post-hoc (off), tamper enforces in-loop (on). Coerce so it can't drift.
        self.attack.enforce_prediction_preserved = self.attack.strand == "tamper"

        return self


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return ExperimentConfig.model_validate(raw)


def load_config_from_dict(data: dict) -> ExperimentConfig:
    """Validate an experiment config already loaded into a dict."""
    return ExperimentConfig.model_validate(data)
