# xai-ids-robustness

A reproducible benchmarking framework for evaluating the **tamper resistance** of
explainable-AI (XAI) methods: SHAP and LIME, in machine-learning-based intrusion
detection systems (IDS).

The framework runs two attack strands under a black-box attack setting:

- **Fragility**: standard FGSM/PGD perturbations (loss objective); explanation
  change is reported for all attacked flows and, separately, for flows whose
  prediction is preserved.
- **Tamper resistance**: a targeted attack (attribution objective) that
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

# with the ML stack 
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

## Final thesis evidence (v4)

The completed primary evidence release contains **126 validated rows**:

- 108 fragility rows: TreeSHAP/XGBoost, LIME/XGBoost, and LIME/MLP across two
  datasets, FGSM-style and PGD-style attacks, three perturbation budgets, and
  three repeatability seeds;
- 18 targeted-tamper rows for the same three model--explainer pairings.

The final analysis input is `results/v4-analysis/`, derived from the preserved
135-row raw record in `results/v4/`. The raw record also contains nine attempted
KernelSHAP/XGBoost backend-control rows. Seven failed the frozen production
repeat-drift gate, so all nine are excluded from robustness and tamper analysis.
They remain audit evidence; they must not be deleted or aggregated with the
primary core. KernelSHAP/MLP was excluded earlier at calibration.

The versioned claim boundary and provenance summary are in
[`docs/v4_results_status.md`](docs/v4_results_status.md) and
[`docs/v4_analysis_manifest.json`](docs/v4_analysis_manifest.json). Large
datasets and result artefacts are intentionally ignored by Git.

## Reproduce or verify the final analysis

```bash
# 1. one-time setup
brew install libomp          # macOS only (XGBoost needs OpenMP)
uv sync --extra ml

# 2. source checks
uv run pytest
uv run ruff check src tests

# 3. final derived analysis input
uv run xai-ids validate-results results/v4-analysis
uv run xai-ids analyze-results results/v4-analysis
```

The stored analysis outputs include within-cohort bootstrap intervals, LIME
attack-versus-noise summaries, and KernelSHAP stability diagnostics. They support
conditional precision statements, not pooled population-level inference.

Every final-analysis row includes frozen-test accuracy, precision, attack-class
recall/F1, class counts, confusion-matrix counts, and
`model_comparison_eligible`. The model-quality rule requires both attack-class
recall and F1 to be at least 0.80; all 126 included rows meet it.

## Re-run the primary grids (expensive)

The following commands generate the **126-row primary core**. They use only the
three eligible model-explainer pairings; do not add KernelSHAP/MLP cells or
alter their axes when reproducing the release. Run them only in a clean working
copy or a new results archive; do not append new rows to the preserved
`results/v4/` record.

```bash
uv run xai-ids grid configs/v4_core_fragility.spec.yaml
uv run xai-ids grid configs/v4_core_tamper.spec.yaml

uv run xai-ids validate-results results/v4 \
  --spec configs/v4_core_fragility.spec.yaml \
  --spec configs/v4_core_tamper.spec.yaml
```

The separate KernelSHAP/XGBoost control was a reliability check, not part of the
primary release. Its incomplete, failed production outcome is documented in the
v4 status record; do not rerun it merely to obtain a more favourable result.
The optional rank-margin sensitivity and synthetic exact-Shapley study remain
future work.

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
