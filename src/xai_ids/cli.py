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

from .config import load_config
from .datasets import build_dataset  # importing the package also registers the loaders
from .explainers import EXPLAINERS as _EXPLAINERS  # noqa: F401 - registers explainers
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


def _cmd_run(args: argparse.Namespace) -> int:
    raise SystemExit(
        "`run` needs the attack/metric modules (Phases 5-6). "
        "Available now: validate, list, prepare, train, explain."
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

    r = sub.add_parser("run", help="run a full experiment (Phases 4-6)")
    r.add_argument("config")
    r.set_defaults(func=_cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
