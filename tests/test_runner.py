"""Tests for runner helpers (slugs, identity, idempotent results table)."""

from __future__ import annotations

import copy
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from xai_ids.config import load_config_from_dict
from xai_ids.interfaces import AttackResult
from xai_ids.results import validate_results_dir
from xai_ids.runner import (
    append_results_row,
    build_per_flow_frame,
    collect_benign_lime_noise,
    collect_kernel_shap_stability,
    experiment_slug,
    results_dir,
    run_experiment,
    run_sweep,
    train_slug,
    true_positive_subset,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg():
    return load_config_from_dict(
        {
            "run": {"seed": 42},
            "dataset": {"name": "unsw_nb15"},
            "model": {"name": "xgboost"},
            "explainer": {"name": "shap", "shap_backend": "tree"},
            "attack": {
                "strand": "tamper",
                "method": "pgd",
                "objective": "attribution",
                "epsilon": 0.03,
                "target_rule": "top1_suppress",
                "target_objective": "suppression",
            },
        }
    )


def test_slugs():
    cfg = _cfg()
    assert train_slug(cfg) == "unsw-xgboost-s42"
    assert experiment_slug(cfg) == "unsw-xgboost-shap-tamp-pgd-e0.03-s42"
    assert results_dir(cfg).as_posix() == "results/v4"

    cfg.explainer.name = "lime"
    cfg.explainer.lime_num_samples = 3000
    assert experiment_slug(cfg) == "unsw-xgboost-lime-tamp-ls3000-pgd-e0.03-s42"

    cfg.attack.target_objective = "rank_margin"
    assert experiment_slug(cfg) == "unsw-xgboost-lime-tamp-ls3000-rm-pgd-e0.03-s42"


def test_true_positive_subset():
    # y_test alternates; model predicts all 1 -> TPs are the y==1 rows.
    bundle = types.SimpleNamespace(
        X_test=np.arange(8).reshape(8, 1).astype(float),
        y_test=np.array([1, 0, 1, 0, 1, 0, 1, 0]),
    )
    model = types.SimpleNamespace(predict=lambda X: np.ones(len(X), dtype=int))
    X_tp, idx = true_positive_subset(model, bundle, size=10, seed=0)
    assert idx.tolist() == [0, 2, 4, 6]  # only true positives
    assert len(X_tp) == 4


def test_append_results_idempotent(tmp_path):
    append_results_row(tmp_path, {"experiment_id": "a", "value": 1})
    append_results_row(tmp_path, {"experiment_id": "a", "value": 2})  # replaces a
    append_results_row(tmp_path, {"experiment_id": "b", "value": 3})
    df = pd.read_csv(tmp_path / "results.csv")
    assert len(df) == 2
    assert int(df.set_index("experiment_id").loc["a", "value"]) == 2
    assert (tmp_path / "manifest.json").exists()


def test_build_per_flow_frame_records_cohort_and_perturbation_evidence():
    cfg = _cfg()
    cfg.metrics.topk = 1
    result = AttackResult(
        X_adv=np.array([[0.2, 0.0], [0.0, 1.2]]),
        prediction_preserved=np.array([True, False]),
        meta={
            "reference_predictions": [1, 1],
            "adversarial_predictions": [1, 0],
            "frozen_due_prediction_flip": [False, True],
            "targets": [0, 1],
            "target_objective": "suppression",
            "target_topk": 1,
            "objective_baseline_attributions": [[1.0, 0.0], [0.0, 1.0]],
        },
    )
    flows = build_per_flow_frame(
        cfg,
        tp_indices=np.array([7, 11]),
        X_base=np.array([[0.0, 0.0], [0.0, 1.0]]),
        result=result,
        adversarial_attributions=np.array([[0.1, 0.9], [0.8, 0.1]]),
    )
    assert flows["test_index"].tolist() == [7, 11]
    assert flows["prediction_preserved"].tolist() == [True, False]
    assert flows["frozen_due_prediction_flip"].tolist() == [False, True]
    assert flows["target_feature"].tolist() == [0.0, 1.0]
    assert flows["target_baseline_rank"].tolist() == [1, 1]
    assert flows["target_adversarial_rank"].tolist() == [2, 2]
    assert flows["target_success_topk_displacement"].tolist() == [True, True]
    assert np.allclose(flows["perturbation_linf"], [0.2, 0.2])
    assert flows["n_changed_features"].tolist() == [1, 1]


def test_collect_benign_lime_noise_uses_fresh_explicit_seeded_explainers(monkeypatch):
    import xai_ids.runner as runner

    cfg = _cfg()
    cfg.explainer.name = "lime"
    cfg.metrics.noise_floor_repeats = 2
    cfg.metrics.noise_floor_seeds = [11, 13]
    created = []

    class _SeededExplainer:
        def __init__(self, seed):
            self.seed = seed

        def explain(self, model, X, background=None):
            return np.asarray(X, dtype=float) + self.seed / 100.0

    def fake_make_explainer(cfg, *, seed=None):
        created.append(seed)
        return _SeededExplainer(seed)

    monkeypatch.setattr(runner, "make_explainer", fake_make_explainer)
    X = np.zeros((2, 2))
    noise = collect_benign_lime_noise(cfg, object(), X, X, X)

    assert created == [11, 13]
    assert noise["mode"] == "repeated_benign_lime"
    assert noise["seeds"].tolist() == [11, 13]
    assert noise["drift"]["l1"].shape == (2, 2)
    assert np.allclose(noise["drift"]["l1"][:, 0], [0.22, 0.26])


def test_collect_kernel_shap_stability_uses_fixed_background_and_fresh_mask_seeds(monkeypatch):
    import xai_ids.runner as runner

    cfg = _cfg()
    cfg.model.name = "mlp"
    cfg.explainer.shap_backend = "kernel"
    cfg.explainer.shap_kernel_stability_repeats = 2
    cfg.explainer.shap_kernel_stability_seeds = [11, 13]
    created = []

    class _KernelExplainer:
        def __init__(self, random_seed):
            self.random_seed = random_seed

        def explain(self, model, X, background=None):
            assert np.array_equal(background, np.ones((3, 2)))
            return np.asarray(X, dtype=float) + self.random_seed / 100.0

        def numerical_diagnostics(self):
            return {
                "calls": 1,
                "attributions_finite": True,
                "max_additivity_error": 0.0,
                "mean_additivity_error": 0.0,
            }

    def fake_make_explainer(cfg, *, seed=None, kernel_random_seed=None):
        created.append((seed, kernel_random_seed))
        return _KernelExplainer(kernel_random_seed)

    monkeypatch.setattr(runner, "make_explainer", fake_make_explainer)
    X = np.zeros((2, 2))
    stability = collect_kernel_shap_stability(cfg, object(), X, X, np.ones((3, 2)))

    assert created == [(42, 11), (42, 13)]
    assert stability["mode"] == "repeated_kernel_shap"
    assert stability["seeds"].tolist() == [11, 13]
    assert stability["cosine_drift"].shape == (2, 2)


def test_grid_kernel_shap_profile_is_applied_only_to_kernel_cells():
    from xai_ids.cli import _expand_spec

    spec = {
        "base": "configs/tamper_example.yaml",
        "kernel_shap": {
            "shap_kernel_l1_reg": 0.0001,
            "shap_kernel_stability_repeats": 3,
            "shap_kernel_stability_seeds": [2003, 2011, 2017],
            "require_shap_kernel_stability": True,
        },
        "axes": {
            "dataset": ["cicids2017"],
            "model": ["xgboost", "mlp"],
            "explainer": ["shap", "lime"],
            "strand": ["tamper"],
            "method": ["pgd"],
            "epsilon": [0.03],
            "seed": [42],
        },
    }
    configs = [load_config_from_dict(value) for value in _expand_spec(spec)]
    kernel = next(
        cfg
        for cfg in configs
        if cfg.explainer.name == "shap" and cfg.explainer.shap_backend == "kernel"
    )
    assert kernel.explainer.require_shap_kernel_stability
    assert kernel.explainer.shap_kernel_stability_repeats == 3
    assert all(
        not cfg.explainer.require_shap_kernel_stability
        for cfg in configs
        if not (cfg.explainer.name == "shap" and cfg.explainer.shap_backend == "kernel")
    )


def test_grid_model_explainer_pairs_exclude_invalid_kernel_shap_mlp_cells():
    from xai_ids.cli import _expand_spec

    spec = {
        "base": "configs/tamper_example.yaml",
        "model_explainer_pairs": [
            {"model": "xgboost", "explainer": "shap", "shap_backend": "tree"},
            {"model": "xgboost", "explainer": "lime"},
            {"model": "mlp", "explainer": "lime"},
        ],
        "axes": {
            "dataset": ["cicids2017", "unsw_nb15"],
            "strand": ["tamper"],
            "method": ["pgd"],
            "epsilon": [0.03],
            "seed": [42],
        },
    }
    configs = [load_config_from_dict(value) for value in _expand_spec(spec)]

    assert len(configs) == 6
    assert {
        (cfg.model.name, cfg.explainer.name, cfg.explainer.shap_backend)
        for cfg in configs
    } == {
        ("xgboost", "shap", "tree"),
        ("xgboost", "lime", "tree"),
        ("mlp", "lime", "tree"),
    }


@pytest.mark.parametrize(
    ("spec_name", "expected_count"),
    [
        ("v4_core_fragility.spec.yaml", 108),
        ("v4_core_tamper.spec.yaml", 18),
        ("v4_kernel_xgboost_control.spec.yaml", 12),
        ("v4_rank_margin.spec.yaml", 18),
        ("v4_core_smoke.spec.yaml", 12),
        ("v4_kernel_xgboost_smoke.spec.yaml", 12),
    ],
)
def test_v4_specs_expand_to_the_predeclared_eligible_cells(spec_name, expected_count):
    from xai_ids.cli import _expand_spec

    spec = yaml.safe_load((CONFIGS / spec_name).read_text(encoding="utf-8"))
    configs = [load_config_from_dict(value) for value in _expand_spec(spec)]

    assert len(configs) == expected_count
    assert all(
        not (cfg.model.name == "mlp" and cfg.explainer.name == "shap")
        for cfg in configs
    )


def test_run_experiment_writes_explicit_cohorts_and_auditable_per_flow_evidence(
    tmp_path, monkeypatch
):
    import xai_ids.runner as runner

    cfg = _cfg()
    cfg.run.output_dir = str(tmp_path)
    cfg.dataset.attack_subset_size = 2
    bundle = types.SimpleNamespace(
        X_train=np.array([[0.0, 0.0], [1.0, 1.0]]),
        X_test=np.array([[0.0, 0.0], [1.0, 1.0]]),
        y_test=np.array([0, 1]),
        meta={"fidelity_reference": {"source": "training"}},
        fidelity_baseline=np.array([0.5, 0.5]),
        feature_groups=[[0], [1]],
    )

    class _Explainer:
        def explain(self, model, X, background=None):
            return np.asarray(X, dtype=float)

    class _Attack:
        def generate(self, model_oracle, X, **kwargs):
            return AttackResult(
                X_adv=np.array([[0.1, 0.0], [0.0, 1.0]]),
                prediction_preserved=np.array([True, False]),
                meta={
                    "targets": [0, 1],
                    "target_objective": "suppression",
                    "target_topk": 10,
                    "objective_baseline_attributions": [[1.0, 0.0], [0.0, 1.0]],
                    "reference_predictions": [1, 1],
                    "adversarial_predictions": [1, 0],
                    "frozen_due_prediction_flip": [False, True],
                    "n_frozen_due_prediction_flip": 1,
                    "model_queries": 17,
                    "explanation_queries": 3,
                    "model_oracle": {"queries": 17},
                    "explanation_oracle": {"queries": 3},
                    "n_perturbed": 1,
                },
            )

    def fake_metrics(*args, require_prediction_preserved, **kwargs):
        n = 1 if require_prediction_preserved else 2
        outcomes = kwargs["tamper_outcomes"]
        preserved = np.asarray(kwargs["preserved"], dtype=bool)
        keep = preserved if require_prediction_preserved else np.ones(len(preserved), dtype=bool)
        metric_values = {
            "n_evaluated": n,
            "retention_rate": 0.5,
            "drift_cosine": 0.25,
            "tamper_success_topk_displacement": float(
                outcomes["success_topk_displacement"][keep].mean()
            ),
        }
        for outcome, values in outcomes.items():
            if outcome == "success_topk_displacement":
                continue
            metric_values[f"target_{outcome}"] = float(np.asarray(values)[keep].mean())
        return metric_values

    monkeypatch.setattr(runner, "build_dataset", lambda cfg: bundle)
    monkeypatch.setattr(runner, "load_or_train_model", lambda cfg, bundle: object())
    monkeypatch.setattr(
        runner,
        "model_quality_record",
        lambda model, bundle: {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "test_n_total": 4,
            "test_n_benign": 2,
            "test_n_attack": 2,
            "true_negative": 2,
            "false_positive": 0,
            "false_negative": 0,
            "true_positive": 2,
            "min_attack_recall": 0.8,
            "min_f1": 0.8,
            "comparison_eligible": True,
        },
    )
    monkeypatch.setattr(
        runner,
        "true_positive_subset",
        lambda model, bundle, size, seed: (np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([4, 8])),
    )
    monkeypatch.setattr(runner, "make_explainer", lambda cfg: _Explainer())
    monkeypatch.setattr(runner, "make_attack", lambda cfg: _Attack())
    monkeypatch.setattr(runner, "make_attack_oracles", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(runner, "compute_metrics", fake_metrics)

    row = run_experiment(cfg)
    assert row["n_total"] == 2
    assert row["n_preserved"] == 1
    assert row["n_flipped"] == 1
    assert row["all_n_evaluated"] == 2
    assert row["preserved_n_evaluated"] == 1
    assert "n_evaluated" not in row
    assert "all_drift_cosine" in row and "preserved_drift_cosine" in row
    assert "all_tamper_success_topk_displacement" in row
    assert row["model_f1"] == 1.0
    assert row["model_comparison_eligible"] is True

    run_dir = tmp_path / "v4" / row["experiment_id"]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {
        "summary",
        "attack_oracle_accounting",
        "noise_floor",
        "model_quality",
        "explainer_warnings",
        "kernel_shap_numerical_stability",
        "all_attacked",
        "prediction_preserved",
    }
    flows = pd.read_csv(run_dir / "per_flow.csv")
    assert flows["prediction_preserved"].tolist() == [True, False]
    assert row["n_explainer_convergence_warnings"] == 0
    assert validate_results_dir(tmp_path / "v4").ok


def test_gen_grid_resolution():
    # fragility -> loss/enforce=False; mlp+shap -> kernel backend
    from copy import deepcopy  # noqa: F401 (documents intent)

    base = copy.deepcopy(
        {
            "attack": {
                "method": "pgd", "epsilon": 0.03, "steps": 50, "step_size": 0.005,
                "spsa": {"q": 2, "delta": 0.01},
            },
        }
    )
    base["dataset"] = {"name": "cicids2017"}
    base["model"] = {"name": "mlp"}
    base["explainer"] = {"name": "shap", "shap_backend": "kernel"}
    base["attack"].update(
        {"strand": "fragility", "objective": "loss",
         "enforce_prediction_preserved": False, "target_rule": "none"}
    )
    cfg = load_config_from_dict(base)
    assert experiment_slug(cfg) == "cicids-mlp-shap-frag-kl10p0001-pgd-e0.03-s42"
    # enforce is coerced from strand: fragility -> False
    assert cfg.attack.enforce_prediction_preserved is False


def test_gen_grid_can_vary_lime_sample_count_without_experiment_id_collisions():
    from xai_ids.cli import _expand_spec

    spec = {
        "base": "unused-in-memory",
        "axes": {
            "dataset": ["cicids2017"],
            "model": ["xgboost"],
            "explainer": ["lime"],
            "strand": ["fragility"],
            "method": ["pgd"],
            "epsilon": [0.03],
            "lime_num_samples": [1500, 3000, 5000],
            "seed": [42],
        },
    }
    base = {
        "dataset": {"name": "cicids2017"},
        "model": {"name": "xgboost"},
        "explainer": {"name": "lime"},
        "attack": {"method": "pgd", "epsilon": 0.03, "steps": 1, "step_size": 0.01},
    }

    from unittest.mock import patch

    with patch("xai_ids.cli.yaml.safe_load", return_value=base), patch(
        "xai_ids.cli.Path.read_text", return_value="base"
    ):
        expanded = _expand_spec(spec)
    cfgs = [load_config_from_dict(item) for item in expanded]
    assert [cfg.explainer.lime_num_samples for cfg in cfgs] == [1500, 3000, 5000]
    assert len({experiment_slug(cfg) for cfg in cfgs}) == 3


def test_gen_grid_can_vary_tamper_target_objective_without_id_collisions():
    from unittest.mock import patch

    from xai_ids.cli import _expand_spec

    spec = {
        "base": "unused-in-memory",
        "axes": {
            "dataset": ["cicids2017"],
            "model": ["xgboost"],
            "explainer": ["shap"],
            "strand": ["tamper"],
            "method": ["pgd"],
            "epsilon": [0.03],
            "target_objective": ["suppression", "rank_margin"],
            "seed": [42],
        },
    }
    base = {
        "dataset": {"name": "cicids2017"},
        "model": {"name": "xgboost"},
        "explainer": {"name": "shap", "shap_backend": "tree"},
        "attack": {"method": "pgd", "epsilon": 0.03, "steps": 1, "step_size": 0.01},
    }
    with patch("xai_ids.cli.yaml.safe_load", return_value=base), patch(
        "xai_ids.cli.Path.read_text", return_value="base"
    ):
        cfgs = [load_config_from_dict(item) for item in _expand_spec(spec)]

    assert [cfg.attack.target_objective for cfg in cfgs] == ["suppression", "rank_margin"]
    assert len({experiment_slug(cfg) for cfg in cfgs}) == 2


def test_gen_grid_can_select_kernel_shap_for_xgboost_without_id_collisions():
    from unittest.mock import patch

    from xai_ids.cli import _expand_spec

    spec = {
        "base": "unused-in-memory",
        "axes": {
            "dataset": ["cicids2017"],
            "model": ["xgboost"],
            "explainer": ["shap"],
            "shap_backend": ["tree", "kernel"],
            "strand": ["fragility"],
            "method": ["pgd"],
            "epsilon": [0.03],
            "seed": [42],
        },
    }
    base = {
        "dataset": {"name": "cicids2017"},
        "model": {"name": "xgboost"},
        "explainer": {"name": "shap", "shap_backend": "tree"},
        "attack": {"method": "pgd", "epsilon": 0.03, "steps": 1, "step_size": 0.01},
    }
    with patch("xai_ids.cli.yaml.safe_load", return_value=base), patch(
        "xai_ids.cli.Path.read_text", return_value="base"
    ):
        cfgs = [load_config_from_dict(item) for item in _expand_spec(spec)]

    assert [cfg.explainer.shap_backend for cfg in cfgs] == ["tree", "kernel"]
    assert [experiment_slug(cfg) for cfg in cfgs] == [
        "cicids-xgboost-shap-frag-pgd-e0.03-s42",
        "cicids-xgboost-shap-frag-skernel-kl10p0001-pgd-e0.03-s42",
    ]


def test_gen_grid_can_vary_kernel_regularisation_without_experiment_id_collisions():
    from unittest.mock import patch

    from xai_ids.cli import _expand_spec

    spec = {
        "base": "unused-in-memory",
        "axes": {
            "dataset": ["cicids2017"],
            "model": ["mlp"],
            "explainer": ["shap"],
            "shap_kernel_l1_reg": [0.001, 0.01],
            "strand": ["fragility"],
            "method": ["pgd"],
            "epsilon": [0.03],
            "seed": [42],
        },
    }
    base = {
        "dataset": {"name": "cicids2017"},
        "model": {"name": "mlp"},
        "explainer": {"name": "shap", "shap_backend": "kernel"},
        "attack": {"method": "pgd", "epsilon": 0.03, "steps": 1, "step_size": 0.01},
    }
    with patch("xai_ids.cli.yaml.safe_load", return_value=base), patch(
        "xai_ids.cli.Path.read_text", return_value="base"
    ):
        cfgs = [load_config_from_dict(item) for item in _expand_spec(spec)]

    assert [cfg.explainer.shap_kernel_l1_reg for cfg in cfgs] == [0.001, 0.01]
    assert [experiment_slug(cfg) for cfg in cfgs] == [
        "cicids-mlp-shap-frag-kl10p001-pgd-e0.03-s42",
        "cicids-mlp-shap-frag-kl10p01-pgd-e0.03-s42",
    ]


def test_synthetic_end_to_end_grid_covers_both_strands_and_explainers(tmp_path, monkeypatch):
    """Phase 8: validate runner artifacts for the 2-strand x 2-explainer matrix."""
    import xai_ids.runner as runner

    bundle = types.SimpleNamespace(
        X_train=np.array([[0.0, 0.0], [1.0, 1.0]]),
        X_test=np.array([[0.0, 0.0], [1.0, 1.0]]),
        y_test=np.array([0, 1]),
        meta={"fidelity_reference": {"source": "training"}},
        fidelity_baseline=np.array([0.5, 0.5]),
        feature_groups=[[0], [1]],
    )

    class _Explainer:
        def explain(self, model, X, background=None):
            return np.asarray(X, dtype=float)

    class _Attack:
        def __init__(self, strand):
            self.strand = strand

        def generate(self, model_oracle, X, **kwargs):
            targets = [0, 1] if self.strand == "tamper" else None
            return AttackResult(
                X_adv=np.array([[0.1, 0.0], [0.0, 1.0]]),
                prediction_preserved=np.array([True, False]),
                meta={
                    "targets": targets,
                    "target_objective": "suppression",
                    "target_topk": 10,
                    "objective_baseline_attributions": (
                        [[1.0, 0.0], [0.0, 1.0]] if targets is not None else None
                    ),
                    "reference_predictions": [1, 1],
                    "adversarial_predictions": [1, 0],
                    "frozen_due_prediction_flip": [False, True],
                    "n_frozen_due_prediction_flip": 1,
                    "model_queries": 17,
                    "explanation_queries": 3 if targets is not None else 0,
                    "model_oracle": {"queries": 17},
                    "explanation_oracle": {"queries": 3} if targets is not None else None,
                    "n_perturbed": 1,
                },
            )

    def fake_metrics(*args, require_prediction_preserved, **kwargs):
        preserved = np.asarray(kwargs["preserved"], dtype=bool)
        keep = preserved if require_prediction_preserved else np.ones(len(preserved), dtype=bool)
        values = {
            "n_evaluated": int(keep.sum()),
            "retention_rate": float(preserved.mean()),
            "drift_cosine": 0.25,
        }
        outcomes = kwargs["tamper_outcomes"]
        if outcomes is not None:
            values["tamper_success_topk_displacement"] = float(
                outcomes["success_topk_displacement"][keep].mean()
            )
            for name, outcome in outcomes.items():
                if name != "success_topk_displacement":
                    values[f"target_{name}"] = float(np.asarray(outcome)[keep].mean())
        return values

    quality = {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "test_n_total": 2,
        "test_n_benign": 1,
        "test_n_attack": 1,
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 1,
        "min_attack_recall": 0.8,
        "min_f1": 0.8,
        "comparison_eligible": True,
    }
    monkeypatch.setattr(runner, "build_dataset", lambda cfg: bundle)
    monkeypatch.setattr(runner, "load_or_train_model", lambda cfg, bundle: object())
    monkeypatch.setattr(runner, "model_quality_record", lambda model, bundle: quality)
    monkeypatch.setattr(
        runner,
        "true_positive_subset",
        lambda model, bundle, size, seed: (
            np.array([[0.0, 0.0], [0.0, 1.0]]),
            np.array([4, 8]),
        ),
    )
    monkeypatch.setattr(runner, "make_explainer", lambda cfg, seed=None: _Explainer())
    monkeypatch.setattr(runner, "make_attack", lambda cfg: _Attack(cfg.attack.strand))
    monkeypatch.setattr(runner, "make_attack_oracles", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(runner, "compute_metrics", fake_metrics)

    cfgs = []
    for strand in ("fragility", "tamper"):
        for explainer in ("shap", "lime"):
            cfg = _cfg()
            cfg.run.output_dir = str(tmp_path)
            cfg.explainer.name = explainer
            cfg.attack.strand = strand
            cfg.attack.objective = "loss" if strand == "fragility" else "attribution"
            cfg.attack.target_rule = "none" if strand == "fragility" else "top1_suppress"
            cfg.attack.enforce_prediction_preserved = strand == "tamper"
            cfgs.append(cfg)

    rows = run_sweep(cfgs)
    assert {(row["strand"], row["explainer"]) for row in rows} == {
        ("fragility", "shap"),
        ("fragility", "lime"),
        ("tamper", "shap"),
        ("tamper", "lime"),
    }
    assert validate_results_dir(tmp_path / "v4").ok
