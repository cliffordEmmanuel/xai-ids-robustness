"""Tests for the experiment config schema, loader, and cross-field validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from xai_ids.config import ExperimentConfig, load_config, load_config_from_dict

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _valid_tamper_dict() -> dict:
    return copy.deepcopy(
        {
            "dataset": {"name": "unsw_nb15"},
            "model": {"name": "xgboost"},
            "explainer": {"name": "shap", "shap_backend": "tree"},
            "attack": {
                "strand": "tamper",
                "method": "pgd",
                "gradient_source": "spsa",
                "objective": "attribution",
                "epsilon": 0.03,
                "steps": 50,
                "step_size": 0.005,
                "target_rule": "top1_suppress",
            },
        }
    )


# --------------------------------------------------------------------------- #
# Valid configs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "fragility_example.yaml",
        "tamper_example.yaml",
        "v4_primary_base.yaml",
        "v4_core_smoke_base.yaml",
        "v4_kernel_xgboost_smoke_base.yaml",
    ],
)
def test_example_configs_load(name: str) -> None:
    cfg = load_config(CONFIGS / name)
    assert isinstance(cfg, ExperimentConfig)


def test_defaults_applied() -> None:
    cfg = load_config_from_dict(_valid_tamper_dict())
    assert cfg.run.seed == 42
    assert cfg.metrics.topk == 10
    assert cfg.explainer.shap_background_size == 100
    assert cfg.explainer.shap_kernel_nsamples == 100
    assert cfg.explainer.shap_kernel_l1_reg == 0.0001
    assert cfg.attack.spsa.q == 2
    assert cfg.threat_model.model_access == "probability_query"
    assert cfg.threat_model.query_budget.explanation_queries == 250


# --------------------------------------------------------------------------- #
# Cross-field validation
# --------------------------------------------------------------------------- #


def test_tamper_requires_attribution_objective() -> None:
    data = _valid_tamper_dict()
    data["attack"]["objective"] = "loss"
    with pytest.raises(ValueError, match="attribution"):
        load_config_from_dict(data)


def test_fragility_requires_loss_objective() -> None:
    data = _valid_tamper_dict()
    data["attack"]["strand"] = "fragility"  # still objective=attribution -> invalid
    with pytest.raises(ValueError, match="loss"):
        load_config_from_dict(data)


def test_autograd_only_for_mlp() -> None:
    data = _valid_tamper_dict()
    data["attack"]["gradient_source"] = "autograd"  # model is xgboost -> invalid
    with pytest.raises(ValueError, match="autograd"):
        load_config_from_dict(data)


def test_tree_shap_requires_xgboost() -> None:
    data = _valid_tamper_dict()
    data["model"]["name"] = "mlp"
    data["attack"]["gradient_source"] = "spsa"
    with pytest.raises(ValueError, match="tree"):
        load_config_from_dict(data)


def test_attribution_requires_target_rule() -> None:
    data = _valid_tamper_dict()
    data["attack"]["target_rule"] = "none"
    with pytest.raises(ValueError, match="target_rule"):
        load_config_from_dict(data)


def test_rank_margin_target_objective_requires_attribution_attack() -> None:
    data = _valid_tamper_dict()
    data["attack"].update(
        {"strand": "fragility", "objective": "loss", "target_objective": "rank_margin"}
    )
    with pytest.raises(ValueError, match="rank_margin"):
        load_config_from_dict(data)


def test_rank_margin_target_objective_is_valid_for_tamper() -> None:
    data = _valid_tamper_dict()
    data["attack"]["target_objective"] = "rank_margin"
    cfg = load_config_from_dict(data)
    assert cfg.attack.target_objective == "rank_margin"


def test_tamper_requires_explanation_oracle() -> None:
    data = _valid_tamper_dict()
    data["threat_model"] = {"explanation_access": "none"}
    with pytest.raises(ValueError, match="explanation_access"):
        load_config_from_dict(data)


def test_noise_floor_seeds_must_be_unique_and_cover_repeats() -> None:
    data = _valid_tamper_dict()
    data["metrics"] = {"noise_floor_repeats": 3, "noise_floor_seeds": [1, 1]}
    with pytest.raises(ValueError, match="noise_floor_seeds"):
        load_config_from_dict(data)


def test_lime_noise_floor_seeds_must_not_reuse_the_main_run_seed() -> None:
    data = _valid_tamper_dict()
    data["explainer"] = {"name": "lime"}
    data["metrics"] = {"noise_floor_repeats": 2, "noise_floor_seeds": [42, 43]}
    with pytest.raises(ValueError, match="main run.seed"):
        load_config_from_dict(data)


def test_unknown_key_rejected() -> None:
    data = _valid_tamper_dict()
    data["attack"]["typo_field"] = 1
    with pytest.raises(ValueError):
        load_config_from_dict(data)


def test_fgsm_coerces_steps_to_one() -> None:
    data = _valid_tamper_dict()
    data["attack"]["strand"] = "fragility"
    data["attack"]["objective"] = "loss"
    data["attack"]["method"] = "fgsm"
    data["attack"]["steps"] = 50
    cfg = load_config_from_dict(data)
    assert cfg.attack.steps == 1


def test_positive_constraints() -> None:
    data = _valid_tamper_dict()
    data["attack"]["epsilon"] = -1.0
    with pytest.raises(ValueError):
        load_config_from_dict(data)


@pytest.mark.parametrize("value", [0, -0.01, "auto", "num_features(0)"])
def test_kernel_shap_regularisation_must_be_explicit_and_valid(value) -> None:
    data = _valid_tamper_dict()
    data["model"]["name"] = "mlp"
    data["explainer"]["shap_backend"] = "kernel"
    data["explainer"]["shap_kernel_l1_reg"] = value
    with pytest.raises(ValueError, match="shap_kernel_l1_reg"):
        load_config_from_dict(data)


def test_required_kernel_stability_needs_independent_repeats() -> None:
    data = _valid_tamper_dict()
    data["model"]["name"] = "mlp"
    data["explainer"].update(
        {
            "shap_backend": "kernel",
            "require_shap_kernel_stability": True,
            "shap_kernel_stability_repeats": 1,
            "shap_kernel_stability_seeds": [7],
        }
    )
    with pytest.raises(ValueError, match="at least two stability repeats"):
        load_config_from_dict(data)
