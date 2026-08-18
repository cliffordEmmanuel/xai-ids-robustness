# xai-ids-robustness

A reproducible benchmarking framework for evaluating the **tamper resistance** of
explainable-AI (XAI) methods — SHAP and LIME — in machine-learning-based intrusion
detection systems (IDS).

The framework runs two attack strands under a black-box attack setting:

- **Fragility** — standard FGSM/PGD perturbations (loss objective); explanation
  change is reported for all attacked flows and, separately, for flows whose
  prediction is preserved.
- **Tamper resistance** — a targeted attack (attribution objective) that
  deliberately redirects the explanation while the prediction is held fixed.

See [`docs/implementation_workplan.md`](docs/implementation_workplan.md) for the
full plan and [`docs/attack_strategy.md`](docs/attack_strategy.md) for the attack
design rationale.

## Install (uv)

This project uses [uv](https://docs.astral.sh/uv/) and targets **Python 3.10–3.12**
(pinned to 3.12 via `.python-version`). Python 3.13+ is not yet supported because
the ML stack's `numba`/`llvmlite` dependencies lack wheels for it. `uv` will fetch
3.12 automatically if it isn't installed. Install uv, then:

```bash
# core skeleton + dev tools (config, interfaces, reproducibility, pytest, ruff)
uv sync

# with the ML stack (from Phase 2 onward)
uv sync --extra ml
```

**System prerequisite (macOS):** XGBoost needs the OpenMP runtime, which is not a
Python package. Install it once with `brew install libomp` (Linux users get it
via `libgomp`, usually already present).

`uv sync` creates `.venv/` and resolves everything into `uv.lock` (committed for
reproducibility). Run commands inside the environment with `uv run`:

```bash
uv run pytest
uv run ruff check src tests
uv run xai-ids validate configs/tamper_example.yaml
```

<details>
<summary>pip fallback</summary>

```bash
pip install -e ".[ml]"    # runtime + ML stack
pip install pytest ruff   # dev tools
```
</details>

## Data

Download the datasets once and place them in `data/` (gitignored):

- CICIDS2017 — https://www.unb.ca/cic/datasets/ids-2017.html → `data/raw/cicids2017/*.csv`
- UNSW-NB15 — https://research.unsw.edu.au/projects/unsw-nb15-dataset → `data/raw/unsw_nb15/UNSW_NB15_{training,testing}-set.csv`

## Run the experiments

The archived `results/v3/` record validates structurally, but its KernelSHAP
cells emitted convergence warnings and are **not** final thesis evidence. Run
the Phase 9 numerical calibration and strict KernelSHAP smoke first; the
replacement evidence directory is `results/v4/`. `results/v1/` through
`results/v3/` are preserved records and are never overwritten.

```bash
# 1. one-time setup
brew install libomp          # macOS only (XGBoost needs OpenMP)
uv sync --extra ml

# 2. calibrate the explicit KernelSHAP numerical setting, then inspect
#    results/kernelshap-calibration-v1/analysis/kernel_shap_candidate_summary.csv
uv run xai-ids grid configs/kernelshap_stability_calibration.spec.yaml
uv run xai-ids validate-results results/kernelshap-calibration-v1 \
  --spec configs/kernelshap_stability_calibration.spec.yaml
uv run xai-ids analyze-results results/kernelshap-calibration-v1

# 3. after applying the selected setting to the release profile, run its strict smoke
uv run xai-ids grid configs/phase9_kernelshap_smoke.spec.yaml
uv run xai-ids validate-results results/smoke-v4-kernelshap \
  --spec configs/phase9_kernelshap_smoke.spec.yaml

# 4. run the two primary v4 grids (each command generates + runs the whole grid)
uv run xai-ids grid configs/gridB.spec.yaml     # tamper resistance  (RQ4)
uv run xai-ids grid configs/gridA.spec.yaml     # fragility          (RQ1-3)
```

Grid B (tamper) is smaller and the headline result — run it first. Grids are
defined by the `configs/*.spec.yaml` files; edit their `axes` to add/remove
datasets, models, explainers, epsilons, or seeds. The LIME and tamper cells are
the slow ones, so start small and scale up if needed.

The focused LIME-noise control is generated/run separately after the main grid:

```bash
uv run xai-ids grid configs/lime_noise_ablation.spec.yaml
```

It holds one representative fragility condition fixed while varying LIME's
samples (1,500 / 3,000 / 5,000) across three run seeds. Each run also stores ten
fresh-seed benign LIME explanations in `noise_floor.npz`.

The Phase 6 rank-margin sensitivity control is separate from the primary tamper
grid, so the two objectives receive distinct experiment IDs and result rows:

```bash
uv run xai-ids grid configs/tamper_rank_margin.spec.yaml
```

For every tamper flow, `per_flow.csv` records the exact attack-time target,
target magnitude and rank before/after, and whether the original top-1 target
left the configured top-*k* set. That final condition is the primary tamper
success measure.

Phase 7 adds two backend-control grids. Run the fragility grid first; the
tamper grid is deliberately separate because KernelSHAP is evaluated in-loop:

```bash
uv run xai-ids grid configs/shap_kernel_xgboost_fragility.spec.yaml
uv run xai-ids grid configs/shap_kernel_xgboost_tamper.spec.yaml
```

Every revised result row includes frozen-test accuracy, precision, attack-class
recall/F1, class counts, confusion-matrix counts, and
`model_comparison_eligible`. Comparative explanation claims require both
attack-class recall and F1 to be at least 0.80; lower-quality model runs remain
in the result set as explicitly labelled evidence.

## Phase 8: smoke, validate, and analyse

Start each release cycle with the small real-data smoke grid. It runs both
strands and both explainers on 12 true-positive CICIDS flows; it is a pipeline
check, not thesis evidence.

```bash
uv run xai-ids grid configs/phase8_smoke.spec.yaml
uv run xai-ids validate-results results/smoke-v3 --spec configs/phase8_smoke.spec.yaml
```

Inspect `metrics.json` for `explainer_warnings`, `per_flow.csv` for tamper/rank
evidence, and `model_comparison_eligible` in `results.csv`. Smoke evidence is
kept in `results/smoke-v3/`, separate from the final `results/v3/` table. If it
is sound, run the planned full grids and validate all their specs together.

After strict v4 validation, create the conditional within-cell analysis artifacts:

```bash
uv run xai-ids analyze-results results/v4
```

This writes `results/v4/analysis/per_flow_bootstrap.csv`,
`results/v4/analysis/lime_attack_vs_noise.csv`, and the KernelSHAP
numerical-stability tables. Bootstrap intervals and paired
sign-flip p-values are conditional within-cell summaries; they are not pooled
independent-condition tests.

For the final release table, validate every planned evidence grid together
before calling `analyze-results`:

```bash
uv run xai-ids validate-results results/v4 \
  --spec configs/gridA.spec.yaml \
  --spec configs/gridB.spec.yaml \
  --spec configs/lime_noise_ablation.spec.yaml \
  --spec configs/tamper_rank_margin.spec.yaml \
  --spec configs/shap_kernel_xgboost_fragility.spec.yaml \
  --spec configs/shap_kernel_xgboost_tamper.spec.yaml
```

See [`docs/kernelshap_stability.md`](docs/kernelshap_stability.md) for the
selection rule and the full Phase 9 release gate.

**Threading / OpenMP (macOS):** XGBoost defaults to a single OpenMP thread here
(`OMP_NUM_THREADS=1`, set automatically on import) because macOS can load two
conflicting OpenMP runtimes — from XGBoost's wheel, other wheels, and Homebrew's
`libomp` — which segfaults during training. This is a safe default, not a hard
limit: once your environment is stable you can opt back into multithreading by
setting the variable yourself before running, e.g. `export OMP_NUM_THREADS=8`.

Long runs: `nohup uv run xai-ids grid configs/gridA.spec.yaml > gridA.log 2>&1 &`
and watch with `tail -f gridA.log`.

## Other commands

```bash
uv run xai-ids validate configs/tamper_example.yaml   # check a config
uv run xai-ids list                                   # registered components
uv run xai-ids run configs/tamper_example.yaml        # one experiment
uv run pytest                                         # test suite
```

## Layout

```
src/xai_ids/
  config.py            # pydantic config models + load_config()
  runner.py            # run_experiment / sweep / grid orchestration
  cli.py               # xai-ids entry point
  reproducibility.py   # seeds, run IDs, run dirs, logging, provenance
  registry.py          # name -> component registry
  interfaces/          # Dataset, Model, Explainer, Attack, Metric ABCs
  datasets/ models/ explainers/ attacks/ metrics/   # the components
configs/               # example configs + grid specs
docs/                  # design + methodology docs
tests/                 # unit tests
```
