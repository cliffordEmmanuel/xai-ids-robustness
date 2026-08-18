"""Command-line entry point for the framework.

Thin layer: parse args, delegate to the library / runner.

    xai-ids validate configs/tamper_example.yaml
    xai-ids list explainers
    xai-ids prepare configs/tamper_example.yaml
    xai-ids train   configs/tamper_example.yaml
    xai-ids explain configs/tamper_example.yaml
    xai-ids run     configs/tamper_example.yaml
    xai-ids grid    configs/gridA.spec.yaml      # run a whole grid (generate + run)
"""

from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .analysis import analyze_results_dir
from .config import load_config, load_config_from_dict
from .datasets import build_dataset
from .registry import ATTACKS, DATASETS, EXPLAINERS, METRICS, MODELS
from .reproducibility import make_run_dir, set_seed, setup_logging, write_provenance
from .results import validate_results_dir
from .runner import (
    experiment_slug,
    explain_slug,
    load_or_train_model,
    make_explainer,
    provenance_extra,
    results_dir,
    run_experiment,
    run_sweep,
    train_slug,
    true_positive_subset,
)

_REGISTRIES = {
    "datasets": DATASETS,
    "models": MODELS,
    "explainers": EXPLAINERS,
    "attacks": ATTACKS,
    "metrics": METRICS,
}


def _cmd_validate(args) -> int:
    cfg = load_config(args.config)
    print(
        f"OK  {args.config}  (dataset={cfg.dataset.name}, model={cfg.model.name}, "
        f"explainer={cfg.explainer.name}, strand={cfg.attack.strand})"
    )
    return 0


def _cmd_list(args) -> int:
    for kind in [args.kind] if args.kind else _REGISTRIES:
        items = _REGISTRIES[kind].available()
        print(f"{kind}: {items if items else '(none yet)'}")
    return 0


def _cmd_prepare(args) -> int:
    cfg = load_config(args.config)
    b = build_dataset(cfg)
    u, c = np.unique(b.y_train, return_counts=True)
    print(
        f"[{cfg.dataset.name}] features={b.n_features} "
        f"train={len(b.y_train):,} test={len(b.y_test):,} "
        f"train_balance={dict(zip(u.tolist(), c.tolist(), strict=True))} "
        f"scaler={type(b.scaler).__name__ if b.scaler else 'none'}"
    )
    return 0


def _cmd_train(args) -> int:
    cfg = load_config(args.config)
    set_seed(cfg.run.seed)
    train_dir = make_run_dir(results_dir(cfg), train_slug(cfg))
    log = setup_logging(train_dir)
    log.info("run dir: %s", train_dir)
    bundle = build_dataset(cfg)
    load_or_train_model(cfg, bundle)  # trains + caches (or loads) model + metrics
    metrics = json.loads((train_dir / "metrics.json").read_text())
    log.info("metrics: %s", metrics)
    return 0


def _cmd_explain(args) -> int:
    cfg = load_config(args.config)
    set_seed(cfg.run.seed)
    run_dir = make_run_dir(results_dir(cfg), explain_slug(cfg))
    log = setup_logging(run_dir)
    log.info("run dir: %s", run_dir)
    bundle = build_dataset(cfg)
    write_provenance(
        run_dir,
        cfg,
        cfg.run.seed,
        extra=provenance_extra(bundle, cfg=cfg, include_fidelity=True),
    )
    model = load_or_train_model(cfg, bundle)
    X_tp, tp_idx = true_positive_subset(
        model, bundle, cfg.dataset.attack_subset_size, cfg.run.seed
    )
    explainer = make_explainer(cfg)
    log.info("baseline %s attributions on %d TP flows ...", cfg.explainer.name, len(X_tp))
    attrs = explainer.explain(model, X_tp, background=bundle.X_train)
    np.savez_compressed(
        run_dir / "baseline_attributions.npz",
        attributions=attrs,
        tp_indices=tp_idx,
    )
    log.info("baseline attributions: shape=%s -> %s", attrs.shape, run_dir)
    return 0


def _cmd_run(args) -> int:
    row = run_experiment(load_config(args.config))
    print(f"done: {row['experiment_id']}")
    return 0


def _cmd_sweep(args) -> int:
    cfgs = [load_config(p) for p in args.configs]
    rows = run_sweep(cfgs)
    print(f"done: {len(rows)} experiments -> results.csv")
    return 0


def _expand_spec(spec: dict) -> list[dict]:
    """Expand a grid spec into a list of experiment config dicts (dependent fields resolved)."""
    base = yaml.safe_load(Path(spec["base"]).read_text())
    axes = spec["axes"]
    kernel_shap = spec.get("kernel_shap", {})
    if not isinstance(kernel_shap, dict):
        raise ValueError("grid spec kernel_shap must be a mapping when supplied")
    configured_pairs = spec.get("model_explainer_pairs")
    if configured_pairs is not None:
        if not isinstance(configured_pairs, list) or not configured_pairs:
            raise ValueError("grid spec model_explainer_pairs must be a non-empty list")
        if "model" in axes or "explainer" in axes:
            raise ValueError(
                "grid spec cannot combine model_explainer_pairs with model/explainer axes"
            )
        pairs: list[dict] = []
        for pair in configured_pairs:
            if not isinstance(pair, dict) or not {"model", "explainer"} <= pair.keys():
                raise ValueError(
                    "each model_explainer_pairs entry must map model and explainer"
                )
            unexpected = set(pair) - {"model", "explainer", "shap_backend"}
            if unexpected:
                raise ValueError(
                    "model_explainer_pairs has unsupported fields: "
                    + ", ".join(sorted(unexpected))
                )
            pairs.append(pair)
    else:
        if "model" not in axes or "explainer" not in axes:
            raise ValueError(
                "grid spec requires model and explainer axes when model_explainer_pairs is absent"
            )
        pairs = [{"model": None, "explainer": None}]
    keys = list(axes)
    configs = []
    for pair, combo in itertools.product(pairs, itertools.product(*[axes[k] for k in keys])):
        vals = dict(zip(keys, combo, strict=True))
        c = deepcopy(base)
        c.setdefault("run", {})["seed"] = vals["seed"]
        c.setdefault("dataset", {})["name"] = vals["dataset"]
        model_name = pair["model"] if configured_pairs is not None else vals["model"]
        explainer_name = pair["explainer"] if configured_pairs is not None else vals["explainer"]
        c.setdefault("model", {})["name"] = model_name
        c.setdefault("explainer", {})["name"] = explainer_name
        if explainer_name == "shap":
            c["explainer"]["shap_backend"] = (
                "tree" if model_name == "xgboost" else "kernel"
            )
        if "shap_backend" in pair:
            c["explainer"]["shap_backend"] = pair["shap_backend"]
        if "shap_backend" in vals:
            c["explainer"]["shap_backend"] = vals["shap_backend"]
        if "shap_background_size" in vals:
            c["explainer"]["shap_background_size"] = vals["shap_background_size"]
        if "shap_kernel_nsamples" in vals:
            c["explainer"]["shap_kernel_nsamples"] = vals["shap_kernel_nsamples"]
        if "shap_kernel_l1_reg" in vals:
            c["explainer"]["shap_kernel_l1_reg"] = vals["shap_kernel_l1_reg"]
        if "shap_kernel_stability_repeats" in vals:
            c["explainer"]["shap_kernel_stability_repeats"] = vals[
                "shap_kernel_stability_repeats"
            ]
        # A mixed SHAP/LIME grid may pre-register a final KernelSHAP release
        # profile without making Kernel-only settings invalid for LIME or
        # TreeSHAP cells.  The profile is applied only after the architecture
        # and backend have been resolved above.
        if (
            c["explainer"]["name"] == "shap"
            and c["explainer"].get("shap_backend") == "kernel"
        ):
            c["explainer"].update(kernel_shap)
        if "lime_num_samples" in vals:
            c["explainer"]["lime_num_samples"] = vals["lime_num_samples"]
        strand = vals["strand"]
        atk = c.setdefault("attack", {})
        atk["strand"] = strand
        atk["objective"] = "loss" if strand == "fragility" else "attribution"
        atk["enforce_prediction_preserved"] = strand == "tamper"
        atk["target_rule"] = "none" if strand == "fragility" else "top1_suppress"
        if "target_objective" in vals:
            atk["target_objective"] = vals["target_objective"]
        atk["epsilon"] = vals["epsilon"]
        if "method" in vals:  # optional axis; else inherit base attack.method
            atk["method"] = vals["method"]
        configs.append(c)
    return configs


def _cmd_gen_grid(args) -> int:
    spec = yaml.safe_load(Path(args.spec).read_text())
    out = Path(spec.get("output_dir", "configs/grid"))
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for c in _expand_spec(spec):
        cfg = load_config_from_dict(c)  # validate
        (out / f"{experiment_slug(cfg)}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        count += 1
    print(f"wrote {count} configs -> {out}")
    return 0


def _cmd_grid(args) -> int:
    """Expand a grid spec in memory and run every experiment into results.csv."""
    spec = yaml.safe_load(Path(args.spec).read_text())
    cfgs = [load_config_from_dict(c) for c in _expand_spec(spec)]
    rows = run_sweep(cfgs)
    print(f"done: {len(rows)} experiments -> results.csv")
    return 0


def _cmd_validate_results(args) -> int:
    expected_ids = None
    if args.specs:
        expected_ids = []
        for spec_path in args.specs:
            spec = yaml.safe_load(Path(spec_path).read_text())
            expected_ids.extend(
                experiment_slug(load_config_from_dict(c)) for c in _expand_spec(spec)
            )
    report = validate_results_dir(args.output_dir, expected_ids=expected_ids)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    if report.ok:
        print(f"OK  {args.output_dir}  ({report.rows} rows)")
        return 0
    for error in report.errors:
        print(f"ERROR: {error}")
    return 1


def _cmd_analyze_results(args) -> int:
    artifacts = analyze_results_dir(
        args.output_dir,
        analysis_dir=args.analysis_dir,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
    )
    print(
        f"OK  {artifacts.output_dir}  "
        f"(bootstrap rows={artifacts.bootstrap_rows}, LIME paired rows={artifacts.lime_test_rows})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xai-ids", description="XAI-IDS robustness benchmark")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a config file")
    v.add_argument("config")
    v.set_defaults(func=_cmd_validate)

    ls = sub.add_parser("list", help="list registered components")
    ls.add_argument("kind", nargs="?", choices=list(_REGISTRIES))
    ls.set_defaults(func=_cmd_list)

    for name, fn, helptext in [
        ("prepare", _cmd_prepare, "load + preprocess a dataset"),
        ("train", _cmd_train, "train + evaluate a model (cached)"),
        ("explain", _cmd_explain, "compute baseline attributions"),
        ("run", _cmd_run, "run one full experiment (dataset->model->explain->attack->metrics)"),
    ]:
        s = sub.add_parser(name, help=f"{helptext} (config-only)")
        s.add_argument("config")
        s.set_defaults(func=fn)

    sw = sub.add_parser("sweep", help="run many experiments into results.csv")
    sw.add_argument("configs", nargs="+", help="config files (globs expanded by the shell)")
    sw.set_defaults(func=_cmd_sweep)

    gr = sub.add_parser("grid", help="run a whole grid from a spec (generate + run)")
    gr.add_argument("spec")
    gr.set_defaults(func=_cmd_grid)

    gg = sub.add_parser("gen-grid", help="write grid config files from a spec (no run)")
    gg.add_argument("spec")
    gg.set_defaults(func=_cmd_gen_grid)

    vr = sub.add_parser("validate-results", help="validate a versioned results directory")
    vr.add_argument("output_dir", help="e.g. results/v2")
    vr.add_argument(
        "--spec",
        dest="specs",
        action="append",
        help="optional grid spec; repeat to validate combined grid coverage",
    )
    vr.set_defaults(func=_cmd_validate_results)

    ar = sub.add_parser(
        "analyze-results",
        help="write conditional bootstrap and paired LIME-noise summaries for validated results",
    )
    ar.add_argument("output_dir", help="completed schema-3.5+ results directory, e.g. results/v4")
    ar.add_argument("--analysis-dir", help="output directory (default: <output_dir>/analysis)")
    ar.add_argument("--bootstrap-repeats", type=int, default=10_000)
    ar.add_argument("--permutation-repeats", type=int, default=10_000)
    ar.add_argument("--seed", type=int, default=2026)
    ar.set_defaults(func=_cmd_analyze_results)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
