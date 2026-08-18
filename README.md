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
cells are not final thesis evidence. Two completed calibration runs found no
KernelSHAP-on-MLP setting stable enough for release. The bounded final v4
release therefore uses TreeSHAP on XGBoost and LIME on both model families, with
a separate strict KernelSHAP-on-XGBoost backend control. `results/v1/` through
`results/v3/` and both calibration directories are preserved and never
overwritten.

```bash
# 1. one-time setup
brew install libomp          # macOS only (XGBoost needs OpenMP)
uv sync --extra ml

# 2. freeze the v4 source/config revision after reviewing the generated grids
uv run pytest
uv run ruff check src tests

# 3. run both real-data release smokes
uv run xai-ids grid configs/v4_core_smoke.spec.yaml
uv run xai-ids validate-results results/smoke-v4-core \
  --spec configs/v4_core_smoke.spec.yaml
uv run xai-ids grid configs/v4_kernel_xgboost_smoke.spec.yaml
uv run xai-ids validate-results results/smoke-v4-kernel-xgboost \
  --spec configs/v4_kernel_xgboost_smoke.spec.yaml

# 4. run the 138-row v4 core release
uv run xai-ids grid configs/v4_core_fragility.spec.yaml
uv run xai-ids grid configs/v4_core_tamper.spec.yaml
uv run xai-ids grid configs/v4_kernel_xgboost_control.spec.yaml
```

The tamper core is smaller than the fragility core, but the release claim needs
all three core grids. The final specs deliberately use fixed
`model_explainer_pairs`; do not add KernelSHAP/MLP cells or alter their axes in
the release cycle. LIME and tamper cells are the slow ones.

Every revised result row includes frozen-test accuracy, precision, attack-class
recall/F1, class counts, confusion-matrix counts, and
`model_comparison_eligible`. Comparative explanation claims require both
attack-class recall and F1 to be at least 0.80; lower-quality model runs remain
in the result set as explicitly labelled evidence.

## Validate and analyse v4

After the core grids complete, validate their combined 138-row table before
analysis. The strict backend smoke/control is a PGD epsilon=0.03 spot-check,
not a full fragility-surface claim.

For the 138-row core release:

```bash
uv run xai-ids validate-results results/v4 \
  --spec configs/v4_core_fragility.spec.yaml \
  --spec configs/v4_core_tamper.spec.yaml \
  --spec configs/v4_kernel_xgboost_control.spec.yaml
```

Then create the conditional within-cell analysis artifacts:

```bash
uv run xai-ids analyze-results results/v4
```

This writes `results/v4/analysis/per_flow_bootstrap.csv`,
`results/v4/analysis/lime_attack_vs_noise.csv`, and the KernelSHAP
numerical-stability tables. Bootstrap intervals and paired sign-flip p-values
are conditional within-cell summaries; they are not pooled
independent-condition tests.

If the core validates and time permits, run the separate 18-row rank-margin
sensitivity grid, then include `configs/v4_rank_margin.spec.yaml` in the final
validation command. See [`docs/v4_completion_workplan.md`](docs/v4_completion_workplan.md)
for the bounded release plan and stop rules.

**Threading / OpenMP (macOS):** XGBoost defaults to a single OpenMP thread here
(`OMP_NUM_THREADS=1`, set automatically on import) because macOS can load two
conflicting OpenMP runtimes — from XGBoost's wheel, other wheels, and Homebrew's
`libomp` — which segfaults during training. This is a safe default, not a hard
limit: once your environment is stable you can opt back into multithreading by
setting the variable yourself before running, e.g. `export OMP_NUM_THREADS=8`.

Long runs: `nohup uv run xai-ids grid configs/v4_core_fragility.spec.yaml > v4_fragility.log 2>&1 &`
and watch with `tail -f v4_fragility.log`.

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
