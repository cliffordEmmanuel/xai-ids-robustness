"""Experiment configuration models and loader.

This module is the single source of truth for the YAML schema described in
``docs/implementation_workplan.md`` (section 2). Configs are parsed and validated
with pydantic v2; cross-field rules encode the Path B design decisions so that an
inconsistent experiment fails fast at load time rather than deep inside a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    output_dir: str = "results/"


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
    lime_num_samples: int = Field(1500, gt=0)
    cache_baseline: bool = True


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
    target_rule: Literal["top1_suppress", "topk_suppress", "none"] = "top1_suppress"
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
    subtract_lime_noise_floor: bool = True
    fixed_cohort: bool = True


# --------------------------------------------------------------------------- #
# Root config + cross-field validation
# --------------------------------------------------------------------------- #


class ExperimentConfig(_Base):
    run: RunConfig = Field(default_factory=RunConfig)
    dataset: DatasetConfig
    model: ModelConfig
    explainer: ExplainerConfig
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

        if errors:
            raise ValueError("Invalid experiment configuration:\n- " + "\n- ".join(errors))

        # Soft normalisation: FGSM is a single step regardless of the configured budget.
        if self.attack.method == "fgsm":
            self.attack.steps = 1

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
