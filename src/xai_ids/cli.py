"""Command-line entry point for the framework.

Thin layer: parse args, delegate to library functions. Subcommands grow as
phases land — ``validate``, ``list``, ``prepare``, ``train`` and ``explain`` work
now; ``run`` arrives with the attack/metric modules (Phases 5-6).

    xai-ids validate configs/tamper_example.yaml
    xai-ids list explainers
    xai-ids prepare configs/tamper_example.yaml
    xai-ids train configs/tamper_example.yaml
    xai-ids explain configs/tamper_example.yaml
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .attacks import ATTACKS as _ATTACKS  # noqa: F401 - registers attacks
from .config import load_config
from .datasets import build_dataset  # importing the package also registers the loaders
from .explainers import EXPLAINERS as _EXPLAINERS  # noqa: F401 - registers explainers
from .metrics import compute_metrics
from .models import evaluate_classification  # importing the package also registers the models
from .registry import ATTACKS, DATASETS, EXPLAINERS, METRICS, MODELS
from .reproducibility import (
    make_run_dir,
    set_seed,
    setup_logging,
    write_provenance,
)

_REGISTRIES = {
    "datasets": DATASETS,
    "models": MODELS,
    "explainers": EXPLAINERS,
    "attacks": ATTACKS,
    "metrics": METRICS,
}

# Short dataset labels for run-directory names.
_DS_ABBR = {"cicids2017": "cicids", "unsw_nb15": "unsw"}


def _train_slug(cfg) -> str:
    """Deterministic run-dir name for training: ``<ds>-<model>-s<seed>``."""
    ds = _DS_ABBR.get(cfg.dataset.name, cfg.dataset.name)
    return f"{ds}-{cfg.model.name}-s{cfg.run.seed}"


def _explain_slug(cfg) -> str:
    """Run-dir name for explanation: ``<ds>-<model>-<explainer>-s<seed>``."""
    ds = _DS_ABBR.get(cfg.dataset.name, cfg.dataset.name)
    return f"{ds}-{cfg.model.name}-{cfg.explainer.name}-s{cfg.run.seed}"


def _make_explainer(cfg):
    """Construct the configured explainer with its relevant hyperparameters."""
    if cfg.explainer.name == "shap":
        return EXPLAINERS.get("shap")(
            backend=cfg.explainer.shap_backend, seed=cfg.run.seed
        )
    return EXPLAINERS.get("lime")(
        num_samples=cfg.explainer.lime_num_samples, seed=cfg.run.seed
    )


_STRAND_ABBR = {"fragility": "frag", "tamper": "tamp"}


def _attack_slug(cfg) -> str:
    ds = _DS_ABBR.get(cfg.dataset.name, cfg.dataset.name)
    strand = _STRAND_ABBR.get(cfg.attack.strand, cfg.attack.strand)
    return (
        f"{ds}-{cfg.model.name}-{cfg.explainer.name}-{strand}"
        f"-e{cfg.attack.epsilon}-s{cfg.run.seed}"
    )


def _make_attack(cfg):
    a = cfg.attack
    return ATTACKS.get(a.method)(
        epsilon=a.epsilon,
        steps=a.steps,
        step_size=a.step_size,
        spsa_q=a.spsa.q,
        spsa_delta=a.spsa.delta,
        target_rule=a.target_rule,
        enforce_prediction_preserved=a.enforce_prediction_preserved,
        seed=cfg.run.seed,
    )


def _true_positive_subset(model, bundle, size, seed):
    """Select flows the model correctly flags as attacks (y==1 and predict==1)."""
    preds = model.predict(bundle.X_test)
    tp = np.where((bundle.y_test == 1) & (preds == 1))[0]
    if len(tp) > size:
        rng = np.random.default_rng(seed)
        tp = np.sort(rng.choice(tp, size, replace=False))
    return bundle.X_test[tp], tp


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(
        f"OK  {args.config}  "
        f"(dataset={cfg.dataset.name}, model={cfg.model.name}, "
        f"explainer={cfg.explainer.name}, strand={cfg.attack.strand})"
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    for kind in [args.kind] if args.kind else _REGISTRIES:
        items = _REGISTRIES[kind].available()
        print(f"{kind}: {items if items else '(none yet)'}")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    b = build_dataset(cfg)
    u, c = np.unique(b.y_train, return_counts=True)
    print(
        f"[{cfg.dataset.name}] features={b.n_features} "
        f"train={len(b.y_train):,} test={len(b.y_test):,} "
        f"attack_subset={len(b.attack_indices)} "
        f"train_balance={dict(zip(u.tolist(), c.tolist(), strict=True))} "
        f"scaler={type(b.scaler).__name__ if b.scaler else 'none'}"
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    set_seed(cfg.run.seed)

    run_dir = make_run_dir(cfg.run.output_dir, _train_slug(cfg))
    log = setup_logging(run_dir)
    log.info("run dir: %s", run_dir)

    log.info("loading + preprocessing dataset '%s' ...", cfg.dataset.name)
    b = build_dataset(cfg)  # resolves cfg.dataset.data_dir to a concrete path
    log.info(
        "dataset ready: features=%d train=%d test=%d attack_subset=%d",
        b.n_features, len(b.y_train), len(b.y_test), len(b.attack_indices),
    )
    write_provenance(run_dir, cfg, cfg.run.seed)  # after build: captures resolved data_dir

    log.info("training %s (class_weight=%s) ...", cfg.model.name, cfg.model.class_weight)
    model = MODELS.get(cfg.model.name)(
        class_weight=cfg.model.class_weight, params=cfg.model.params
    )
    model.fit(b.X_train, b.y_train)

    log.info("evaluating on frozen test set ...")
    metrics = evaluate_classification(model, b.X_test, b.y_test)
    log.info(
        "metrics: %s", " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )

    model.save(run_dir / f"model{model.ext}")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("saved model + metrics -> %s", run_dir)
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    set_seed(cfg.run.seed)

    run_dir = make_run_dir(cfg.run.output_dir, _explain_slug(cfg))
    log = setup_logging(run_dir)
    log.info("run dir: %s", run_dir)

    log.info("loading + preprocessing dataset '%s' ...", cfg.dataset.name)
    b = build_dataset(cfg)
    write_provenance(run_dir, cfg, cfg.run.seed)

    log.info("training %s (in-memory) ...", cfg.model.name)
    model = MODELS.get(cfg.model.name)(
        class_weight=cfg.model.class_weight, params=cfg.model.params
    )
    model.fit(b.X_train, b.y_train)

    log.info("computing %s baseline attributions on the attack subset ...", cfg.explainer.name)
    explainer = _make_explainer(cfg)
    attrs = explainer.explain(model, b.X_attack, background=b.X_train)

    out = run_dir / "baseline_attributions.npz"
    np.savez_compressed(out, attributions=attrs, attack_indices=b.attack_indices)
    log.info("baseline attributions: shape=%s -> %s", attrs.shape, out)
    return 0


def _cmd_attack(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    set_seed(cfg.run.seed)

    run_dir = make_run_dir(cfg.run.output_dir, _attack_slug(cfg))
    log = setup_logging(run_dir)
    log.info("run dir: %s", run_dir)

    log.info("loading + preprocessing dataset '%s' ...", cfg.dataset.name)
    b = build_dataset(cfg)
    write_provenance(run_dir, cfg, cfg.run.seed)

    log.info("training %s (in-memory) ...", cfg.model.name)
    model = MODELS.get(cfg.model.name)(
        class_weight=cfg.model.class_weight, params=cfg.model.params
    )
    model.fit(b.X_train, b.y_train)

    X_tp, tp_idx = _true_positive_subset(
        model, b, cfg.dataset.attack_subset_size, cfg.run.seed
    )
    log.info("true-positive attack subset: %d flows", len(X_tp))

    explainer = _make_explainer(cfg)
    log.info("baseline %s attributions ...", cfg.explainer.name)
    baseline = explainer.explain(model, X_tp, background=b.X_train)

    log.info(
        "running %s attack (strand=%s, objective=%s, eps=%s, steps=%d) ...",
        cfg.attack.method, cfg.attack.strand, cfg.attack.objective,
        cfg.attack.epsilon, cfg.attack.steps,
    )
    attack = _make_attack(cfg)
    result = attack.generate(
        model,
        X_tp,
        objective=cfg.attack.objective,
        gradient_source=cfg.attack.gradient_source,
        explainer=explainer if cfg.attack.strand == "tamper" else None,
    )
    adversarial = explainer.explain(model, result.X_adv)

    np.savez_compressed(
        run_dir / "attack.npz",
        X_adv=result.X_adv,
        tp_indices=tp_idx,
        baseline_attributions=baseline,
        adversarial_attributions=adversarial,
        prediction_preserved=result.prediction_preserved,
    )

    log.info("computing metrics ...")
    metrics = compute_metrics(
        baseline,
        adversarial,
        X_base=X_tp,
        X_adv=result.X_adv,
        model=model,
        preserved=result.prediction_preserved,
        metrics_cfg=cfg.metrics,
        explainer=explainer,
        targets=result.meta["targets"],
    )
    metrics["queries"] = result.n_queries
    metrics["n_perturbed"] = result.meta["n_perturbed"]
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log.info(
        "retention=%.3f perturbed=%d/%d queries=%d",
        result.retention_rate, result.meta["n_perturbed"], len(X_tp), result.n_queries,
    )
    _keys = ("drift", "spearman", "topk", "tamper")
    _summary = {k: v for k, v in metrics.items() if k.startswith(_keys)}
    log.info("metrics: %s", _summary)
    log.info("saved -> %s", run_dir)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    raise SystemExit(
        "`run` needs the metric module (Phase 6). "
        "Available now: validate, list, prepare, train, explain, attack."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xai-ids", description="XAI-IDS robustness benchmark")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a config file")
    v.add_argument("config")
    v.set_defaults(func=_cmd_validate)

    ls = sub.add_parser("list", help="list registered components")
    ls.add_argument("kind", nargs="?", choices=list(_REGISTRIES))
    ls.set_defaults(func=_cmd_list)

    pr = sub.add_parser("prepare", help="load + preprocess a dataset (config-only)")
    pr.add_argument("config")
    pr.set_defaults(func=_cmd_prepare)

    tr = sub.add_parser("train", help="train + evaluate a model (config-only)")
    tr.add_argument("config")
    tr.set_defaults(func=_cmd_train)

    ex = sub.add_parser("explain", help="compute baseline attributions (config-only)")
    ex.add_argument("config")
    ex.set_defaults(func=_cmd_explain)

    at = sub.add_parser("attack", help="run an adversarial attack (config-only)")
    at.add_argument("config")
    at.set_defaults(func=_cmd_attack)

    r = sub.add_parser("run", help="run a full experiment (Phases 4-6)")
    r.add_argument("config")
    r.set_defaults(func=_cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
