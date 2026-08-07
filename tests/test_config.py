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


@pytest.mark.parametrize("name", ["fragility_example.yaml", "tamper_example.yaml"])
def test_example_configs_load(name: str) -> None:
    cfg = load_config(CONFIGS / name)
    assert isinstance(cfg, ExperimentConfig)


def test_defaults_applied() -> None:
    cfg = load_config_from_dict(_valid_tamper_dict())
    assert cfg.run.seed == 42
    assert cfg.metrics.topk == 10
    assert cfg.attack.spsa.q == 2


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
