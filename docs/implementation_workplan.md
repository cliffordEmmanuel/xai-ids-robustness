# MSc Dissertation — Implementation Workplan (v2, Path B)

**Supersedes:** `MSc_Dissertation_Implementation_Work_Plan.md` (v1).
**Companion:** `docs/attack_strategy.md` (attack design rationale).
**Change in one line:** the framework now runs **two attack strands** — a broad
*fragility* study (standard FGSM/PGD) and a bounded *tamper-resistance* study
(targeted SPSA attack) — under a single black-box, prediction-preserving threat
model. Everything below is organised so it can be implemented against a settled
spec.

---

## 0. What changed from v1 (read this first)

| # | Decision | Effect on the plan |
|---|----------|--------------------|
| 1 | **Attacks are black-box via SPSA** (zeroth-order), not surrogate transfer or autograd. | One uniform attack on both models; XGBoost's non-differentiability is no longer a special case. |
| 2 | **Two strands.** Fragility (loss objective) + Tamper resistance (attribution objective). | New config fields, a second runner path, and split reporting in metrics/analysis. |
| 3 | **Targeted attack** optimises the explanation itself, prediction held fixed. | `Attack` depends on `Explainer`; constraint enforced in-loop. |
| 4 | **Tamper study = 8 cells** (2 datasets × 2 models × 2 explainers). | Two execution grids instead of one. |
| 5 | **Safeguards**: fixed cohort, objective/eval separation, LIME noise floor. | New metric-engine logic. |

The research **objective is unchanged**: evaluate the tamper resistance of SHAP
and LIME in ML-based IDS. The two strands map to the thesis as:

- **Fragility** → Objectives 1–3 / RQ1–RQ3 (how much explanations degrade; does
  architecture moderate it).
- **Tamper resistance** → Objective 4 / RQ4 (can an adversary deliberately
  redirect the explanation while the prediction holds; does architecture moderate
  *that*).

---

## 1. Terminology (fixed definitions)

- **Strand** — one of `fragility` or `tamper`.
- **Fragility attack** — FGSM or PGD whose objective is the model's **loss**.
  Prediction-preserving samples are *retained* after the fact.
- **Tamper attack** — targeted PGD whose objective is the **explanation** (top-1
  attribution suppression). Prediction preservation is *enforced during*
  optimisation.
- **Gradient source** — `spsa` (default, both models) or `autograd` (MLP only,
  optional sanity check).
- **Attack subset** — the small, frozen slice of the test set (~300–500 samples
  per cell) used for the expensive tamper strand.
- **Cell** — one (dataset, model, explainer) combination.

---

## 2. Configuration schema (single source of truth)

Experiments are fully driven by YAML. The schema below is the contract every
module reads.

```yaml
run:
  id: null                 # auto-generated if null (timestamp + hash)
  seed: 42
  seeds: [42, 7, 123]      # multi-seed repeats
  output_dir: results/

dataset:
  name: cicids2017         # {cicids2017, unsw_nb15}
  normalise: standard      # scaler is fitted on TRAIN, frozen, persisted
  balance: smote_train_only # NEVER touches test
  attack_subset_size: 500  # frozen slice used by the tamper strand

model:
  name: xgboost            # {xgboost, mlp}
  # both models MUST expose predict_proba (the attack surface)

explainer:
  name: shap               # {shap, lime}
  shap_backend: tree       # tree (XGBoost) | kernel (MLP/agnostic)
  lime_num_samples: 1500   # capped for in-loop cost
  cache_baseline: true

attack:
  strand: fragility        # {fragility, tamper}
  method: pgd              # {fgsm, pgd}
  gradient_source: spsa    # {spsa, autograd}
  objective: loss          # {loss, attribution}   # attribution ⇒ tamper strand
  epsilon: 0.03            # L-inf ball, NORMALISED feature space
  steps: 50                # PGD step budget (bounded; early-stop before flip)
  step_size: 0.005
  spsa:
    q: 2                   # perturbation vectors averaged per step (antithetic)
    delta: 0.01            # SPSA finite-difference magnitude
  target_rule: top1_suppress   # tamper only: suppress top-1 baseline feature
  enforce_prediction_preserved: true

metrics:
  stability: [spearman, topk_jaccard]
  fidelity:  [local_fidelity, deletion, insertion]  # need model access
  drift:     [cosine, l1, l2]
  topk: 10                 # fixed (or fraction) for cross-dataset comparability
  subtract_lime_noise_floor: true
  fixed_cohort: true       # samples preserved across ALL epsilon
```

**Cross-dataset note:** `epsilon` is comparable *within* a dataset only. Report
per dataset; do not equate ε values across CICIDS2017 and UNSW-NB15.

---

## 3. Phases

Each phase lists **Goal**, **Tasks**, **Deliverable**, and **Changed vs v1** so
the delta is explicit.

### Phase 1 — Framework Foundation

**Goal:** build the reusable skeleton before any experiment.

**Tasks**
- Repository layout: `src/`, `configs/`, `scripts/`, `tests/`, `results/`,
  `notebooks/`, `docs/`.
- YAML config loader + validation against the schema in §2.
- Abstract interfaces:
  - `Dataset.load()`
  - `Model.fit() / predict() / predict_proba() / save() / load()`
  - `Explainer.explain(model, X) -> np.ndarray`
  - `Attack.generate(model, X, *, objective, gradient_source, explainer=None)`
  - `Metric.compute(baseline, adversarial, *, model=None)`
- Reproducibility: fixed seeds (incl. SPSA/attack seeds), logging, run IDs,
  timestamped outputs, version pinning.

**Deliverable:** benchmarking skeleton with the two-strand-aware interfaces.

**Changed vs v1:** `Attack.generate` now takes a pluggable `objective`, a
`gradient_source`, and an optional `explainer` (needed for the tamper strand).
`Metric.compute` gains optional `model` access (for fidelity). These two signature
changes are the core architectural difference — get them right here.

---

### Phase 2 — Dataset Integration

**Goal:** benchmark datasets in one consistent, frozen format.

**Tasks**
- Loaders for CICIDS2017 and UNSW-NB15.
- Clean: remove duplicates; handle NaN/Inf; encode labels.
- **Normalise** features; **fit scaler on train only, freeze and persist it.**
- Train/test split; **balance train only** (never test).
- Persist processed datasets; **freeze the test set**.
- Carve a frozen **attack subset** (`attack_subset_size`) from the test set.
- Build a **feature-index map** so the tamper target rule is dataset-agnostic.

**Deliverable:** reusable dataset module + frozen test sets + attack subsets.

**Changed vs v1:** scaler must be persisted (ε lives in normalised space); add the
attack subset and the feature-index map.

---

### Phase 3 — IDS Model Integration

**Goal:** integrate XGBoost and MLP behind one interface.

**Tasks**
- Implement XGBoost and MLP with `fit/predict/predict_proba/save/load`.
- Evaluate Accuracy, Precision, Recall, F1; persist trained models.
- Guarantee **`predict_proba` is the black-box attack surface** for both.

**Deliverable:** reusable model module + persisted models + performance table.

**Changed vs v1:** emphasis on `predict_proba` as the uniform attack entry point;
the MLP need not expose gradients (SPSA is used for both), though `autograd` may
be wired as an optional MLP-only sanity check.

---

### Phase 4 — XAI Module

**Goal:** standardised, fast explanation generation.

**Tasks**
- Integrate SHAP (TreeSHAP for XGBoost, KernelSHAP for MLP) and LIME.
- `explain(model, X) -> np.ndarray`; cache baseline attributions.
- **Performance for in-loop use:** cap LIME `num_samples`, fix LIME seed.
- **Target selector:** derive the top-1 attributed feature per sample (tamper
  target rule).
- **LIME noise-floor utility:** repeat LIME N times on the *unperturbed* input to
  estimate intrinsic variance.

**Deliverable:** reusable explanation engine + target selector + noise-floor tool.

**Changed vs v1:** explanations now run inside the attack loop, so the module adds
the target selector and the noise-floor estimator, and must be fast.

---

### Phase 5 — Adversarial Attack Module  *(largest change)*

**Goal:** one attack framework covering both strands and both models.

**Tasks**
- **SPSA gradient estimator** on `predict_proba` (antithetic, `q` vectors,
  `delta`).
- **Unified PGD** with pluggable `gradient_source` (`spsa` default, `autograd`
  MLP-only); **FGSM** = single-step PGD.
- **Objective = `loss`** (fragility strand): standard loss-maximising attack;
  retain prediction-preserved samples afterwards.
- **Objective = `attribution`** (tamper strand): targeted suppression of the
  top-1 baseline feature; **prediction preservation enforced in-loop** via
  penalty/projection; the attack **calls the Explainer** each step.
- **Bounded perturbation:** fixed `steps`, early-stop before a prediction flip
  (explore the ε-ball, don't race to the boundary).
- *(Optional)* Kantchelian exact tree-evasion module for a small validation
  spot-check of SPSA attack strength.

**Deliverable:** reusable attack engine supporting `fragility` and `tamper`.

**Changed vs v1:** replaces "FGSM/PGD + resolve XGBoost strategy." Adds SPSA, the
attribution objective, the in-loop prediction constraint, and Explainer coupling.
Effectively ~2× the v1 scope for this phase.

---

### Phase 6 — Benchmark Metric Engine

**Goal:** reusable robustness metrics with the required safeguards.

**Tasks**
- Stability: Spearman, top-k Jaccard (fixed/fractional `k`).
- Fidelity: local fidelity, deletion, insertion (**require model access**).
- Drift: cosine, L1, L2 — reported as **incidental** (fragility) vs **targeted**
  (tamper).
- **Objective/evaluation separation:** exclude the attack's objective metric from
  the reported tamper metrics (no teaching-to-the-test).
- **Fixed evaluation cohort:** samples prediction-preserved across *all* ε;
  report **retention rate** separately.
- **Subtract the LIME noise floor** from measured drift.

**Deliverable:** reusable metric engine with safeguards baked in.

**Changed vs v1:** adds incidental/targeted split, objective/eval separation,
fixed-cohort + retention rate, noise-floor subtraction, and model access for
fidelity.

---

### Phase 7 — Benchmark Runner

**Goal:** orchestrate the full lifecycle from config.

**Tasks**
- **Fragility path (linear):** load → model → baseline explain → adversarial
  examples → adversarial explain → metrics → store → plot.
- **Tamper path:** load → model → baseline explain (+ target select) →
  targeted attack *(explainer in-loop)* → adversarial explain → metrics →
  store → plot.
- Strand selected by config; one **row per experiment** with full provenance
  (see §4 columns).

**Deliverable:** configurable benchmarking engine with two strand paths.

**Changed vs v1:** the single linear pipeline becomes two paths; the tamper path
breaks linearity because the attack invokes the explainer.

---

### Phase 8 — Benchmark Execution

**Goal:** run both grids.

**Grid A — Fragility (full, cheap):**
```
{cicids2017, unsw_nb15} × {xgboost, mlp} × {shap, lime}
        × {fgsm, pgd} × epsilon-sweep × seeds
```

**Grid B — Tamper resistance (8 cells, bounded):**
```
{cicids2017, unsw_nb15} × {xgboost, mlp} × {shap, lime} × {targeted}
        ~300–500 samples/cell, modest epsilon-sweep + seeds
```

**Master results table columns (min):**
`run_id, dataset, model, explainer, strand, method, gradient_source,
attack_objective, epsilon, seed, target_rule, retention_rate,
incidental_drift, targeted_drift, stability_*, fidelity_*, noise_floor,
tamper_success`

**Deliverable:** benchmark database (one row per experiment, both grids).

**Changed vs v1:** two grids; expanded results schema.

---

### Phase 9 — Analysis and Visualisation

**Goal:** answer RQ1–RQ4 with figures and statistics.

**Tasks**
- **Fragility (RQ1–3):** drift vs ε, stability vs ε, fidelity vs ε; SHAP vs LIME;
  XGBoost vs MLP; dataset comparisons; retention-rate curves.
- **Tamper (RQ4):** targeted drift vs ε; **tamper success rate** (top-1 feature
  suppressed while prediction preserved); SHAP vs LIME and XGBoost vs MLP tamper
  resistance.
- Statistical significance testing within each strand; keep incidental vs targeted
  clearly separated in every plot.

**Deliverable:** publication-ready figures and tables.

**Changed vs v1:** adds the tamper-strand analyses and the tamper success metric.

---

### Phase 10 — Packaging and Reproducibility

**Goal:** ship a reproducible framework.

**Tasks**
- README, install guide, example configs (one per strand), reproduction
  instructions, `CITATION.cff`, license, API docs.
- Reference `docs/attack_strategy.md`; document the two strands, SPSA, the config
  schema, and the 8-cell tamper scope.
- Demonstrate reproducibility via **config-only** changes across both grids.

**Deliverable:** packaged, documented, reproducible framework.

---

## 4. Final architecture

```
Benchmark Framework
│
├── Dataset Module          (loaders, frozen scaler, attack subset, feature map)
├── Model Module            (xgboost, mlp; predict_proba = attack surface)
├── Explainer Module        (SHAP/LIME, target selector, LIME noise floor)
├── Attack Module           (SPSA-PGD; objectives: loss | attribution)
│        └── depends on Explainer for the tamper strand
├── Metric Engine           (stability, fidelity, drift; safeguards)
├── Benchmark Runner        (fragility path | tamper path)
├── Experiment Definitions  (YAML per §2)
├── Results Database        (one row/experiment; both grids)
└── Analysis & Visualisation
```

---

## 5. Compute plan (Apple M4 Max, ~14–16 cores, 48 GB)

CPU-parallel, embarrassingly parallel across samples; no CUDA needed.
`queries = cells × samples/cell × steps × (2 × q)`. TreeSHAP cells are near-free;
KernelSHAP/LIME cells run in minutes–tens-of-minutes across ~12 workers. **The full
8-cell tamper study is an overnight (~3–4 h) job.** Keep `lime_num_samples` and
`steps × q` modest — those are the two knobs that cause blow-ups. Colab Pro does
not help (CPU-starved VMs); use only as optional overflow.

---

## 6. Expected outputs (unchanged in spirit)

1. **Engineering:** modular, config-driven benchmarking framework.
2. **Empirical:** fragility grid + 8-cell tamper-resistance results.
3. **Scientific:** architecture-aware robustness *and* tamper-resistance analysis
   with practitioner recommendations.
