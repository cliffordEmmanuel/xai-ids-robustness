# xai-ids-robustness

A reproducible benchmarking framework for evaluating the **tamper resistance** of
explainable-AI (XAI) methods — SHAP and LIME — in machine-learning-based intrusion
detection systems (IDS).

The framework runs two attack strands under one black-box, prediction-preserving
threat model:

- **Fragility** — standard FGSM/PGD perturbations (loss objective); how much
  explanations degrade under adversarial input change.
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

`uv sync` creates `.venv/` and resolves everything into `uv.lock` (committed for
reproducibility). Run commands inside the environment with `uv run`:

```bash
uv run pytest
uv run ruff check src tests scripts
uv run python scripts/validate_config.py configs/fragility_example.yaml
```

<details>
<summary>pip fallback</summary>

```bash
pip install -e ".[ml]"    # runtime + ML stack
pip install pytest ruff   # dev tools
```
</details>

## Layout

```
src/xai_ids/
  config.py            # pydantic config models + load_config()
  reproducibility.py   # seeds, run IDs, run dirs, logging, provenance
  registry.py          # name -> component registry
  interfaces/          # Dataset, Model, Explainer, Attack, Metric ABCs
configs/               # example experiment configs (fragility, tamper)
scripts/               # CLI helpers (e.g. config validation)
tests/                 # unit tests
```

## Data

Datasets are downloaded manually and placed in `data/` (gitignored):

- CICIDS2017 — https://www.unb.ca/cic/datasets/ids-2017.html → `data/raw/cicids2017/*.csv`
- UNSW-NB15 — https://research.unsw.edu.au/projects/unsw-nb15-dataset → `data/raw/unsw_nb15/UNSW_NB15_{training,testing}-set.csv`

Loaders take the folder path and read whatever CSVs are there. Both datasets are free for academic use and are cited.

## CLI

Single entry point (`xai-ids`). Subcommands grow as phases land:

```bash
uv run xai-ids validate configs/tamper_example.yaml   # validate a config
uv run xai-ids list datasets                           # list registered components
uv run xai-ids prepare configs/tamper_example.yaml     # load + preprocess
uv run xai-ids train configs/tamper_example.yaml       # train + evaluate a model (needs [ml])
uv run xai-ids run configs/tamper_example.yaml         # full experiment (Phases 4-6)
```

## Quick check

```bash
uv run pytest
```
