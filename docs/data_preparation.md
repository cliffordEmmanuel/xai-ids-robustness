# Data Preparation & Preprocessing

How each dataset is turned into a frozen, model-ready `DataBundle`, and **why**
each step is done the way it is. This reflects the implemented code:
`src/xai_ids/datasets/` (`cicids2017.py`, `unsw_nb15.py`, `base.py`,
`preprocess.py`). See also `data_dictionary.md` and `attack_signatures.md`.

## Design: two stages

The pipeline is split into a **per-dataset loader** and a **shared
preprocessor**, on purpose:

- **Loader (`load_raw`)** does the *dataset-specific* work — reading the files
  and cleaning their quirks — and returns a `RawData` (a cleaned feature table +
  binary labels).
- **Preprocessor (`preprocess`)** does the *identical, dataset-agnostic* work —
  split, scale, balance, subset — for both datasets.

Why separate them: the two datasets differ only in *how they're read and
cleaned*; everything after that must be **the same** for a fair comparison.
Keeping the shared steps in one place guarantees CICIDS2017 and UNSW-NB15 are
treated identically downstream, so any difference in results is about the data or
the model — not the pipeline.

## Ordering (and why it matters)

1. **Clean** (in the loader) — happens on the whole dataset *before* the split.
   These operations are per-row and deterministic (drop a leaky column, map a
   label, drop a NaN row), so doing them pre-split introduces no leakage.
2. **Split** — stratified train/test.
3. **Scale & balance** (in the preprocessor) — happen *after* the split and use
   **train only**. This is the critical ordering rule: anything that *learns*
   from the data (the scaler's mean/variance, synthetic balancing) must never see
   the test set, or results are optimistically biased.

---

## Stage 1 — Loading & cleaning (per dataset)

### CICIDS2017 (`CICIDS2017Loader`)

1. **Read & concatenate** all 8 `MachineLearningCVE` CSVs into one table.
   *Why:* the attack types are split across day/session files; the model needs
   them together.
2. **Strip whitespace from column names.** *Why:* the released CSVs ship with
   leading spaces (e.g. `' Label'`), which would otherwise break column lookups.
3. **Binarise the label** — `BENIGN → 0`, any attack name → `1` (original names
   kept in `meta`). *Why:* the study is binary attack-vs-normal detection; the
   attack family isn't the target.
4. **Drop leaky identifier columns** — Flow ID, source/destination IP, source
   port, **Destination Port** (default), Timestamp, and the duplicated
   `Fwd Header Length.1`. *Why:* these encode *who/where*, not attack *behaviour*.
   Destination Port in particular almost perfectly identifies the attacked
   service, so keeping it lets the model "cheat" — a well-known CICIDS pitfall.
   (Toggle with `drop_destination_port=False`.)
5. **Coerce to numeric, replace ±inf with NaN, drop rows with any NaN.**
   *Why:* `Flow Bytes/s` and `Flow Packets/s` contain infinities/NaN from
   divide-by-zero durations; these are a small fraction of rows and cannot be fed
   to a model.
6. **Drop duplicate rows** (features + label together). *Why:* CICIDS contains
   exact duplicates that would otherwise inflate counts and bias training.

Result: ~76 numeric features, binary label.

### UNSW-NB15 (`UNSWNB15Loader`)

1. **Read & concatenate** the pre-partitioned `training-set` and `testing-set`
   CSVs. *Why:* we re-split ourselves (see Stage 2) so both datasets go through
   the identical split logic; the official partition is just the source data.
2. **Normalise column names** (strip + lowercase). *Why:* consistent lookups.
3. **Binarise** using the existing `label` (0/1); keep `attack_cat` only in
   `meta`. *Why:* same binary framing as CICIDS.
4. **Drop `id`, `attack_cat`, `label`** from the features. *Why:* `id` is a row
   index (leakage), `attack_cat` is a second label, `label` is the target.
5. **Retain `proto`, `service`, and `state` as categorical fields.** Their
   rare-category grouping and one-hot encoding are deferred until after the
   train/test split. This prevents test-only category frequencies or values from
   influencing the training representation.
6. **Coerce numeric fields, handle NaN/inf, drop duplicates** — as above.

Result: ~45 base fields expanded by one-hot into numeric features, binary label.

---

## Stage 2 — Shared preprocessing (`preprocess`)

Identical for both datasets. Inputs come from `dataset.*` config plus
`run.seed`.

1. **Stratified train/test split** (`test_size`, seeded).
   *Why stratified:* both datasets are imbalanced; stratifying preserves the
   attack ratio in both splits so the test set stays representative.
   *Why our own split:* running both datasets through the same split keeps them
   comparable and lets us freeze a single test set per run.
2. **Fit categorical encoding on train only, then transform both.** For UNSW,
   the top categories and one-hot output columns are learned from training rows;
   unseen or rare test values map to `other`. *Why:* fitting them before the
   split would leak test-distribution information.
3. **Convert to numeric matrix, fit the scaler on train only, then transform
   both** (StandardScaler by
   default; MinMax or none available). *Why train-only:* fitting on all data
   leaks test statistics into training. *Why scale at all:* it puts features on a
   common scale — needed for the MLP, for meaningful distances in LIME/SHAP, and
   so the attack's `epsilon` means the same thing across features. **`epsilon`
   therefore lives in normalised space** and is only comparable *within* a
   dataset.
4. **Balance the training set only** (`none` / `undersample_train_only` /
   `smote_train_only`). *Why train-only:* the **test set must stay untouched and
   realistically imbalanced** to measure real performance; balancing only helps
   the model learn the minority (attack) class during training.
5. **Carve a frozen attack subset** from the test set (seeded random sample of
   `attack_subset_size`, default 500). *Why:* the tamper strand runs an expensive
   per-sample optimisation, so it operates on a fixed, manageable slice rather
   than the whole test set — and freezing it (by seed) keeps runs comparable.
6. **Return a `DataBundle`** carrying train/test arrays, the frozen attack subset
   + indices, the fitted scaler, feature names, and meta.

The result is a `feature_index` map (name → column) on the bundle, used later by
the tamper target rule to name/locate the suppressed feature.

---

## Persistence

`save_bundle` / `load_bundle` write the arrays (`arrays.npz`), the fitted scaler
(`scaler.joblib`), any fitted categorical transformer
(`categorical_transformer.joblib`), and `meta.json`. *Why:* preprocessing runs
once per config; persisting the **frozen** split and all train-fitted transforms
makes every downstream step (models, explanations, attacks) reproducible and
provenance-traceable.

## Fidelity reference input

Deletion and insertion fidelity use one fixed reference vector derived from the
training split for both the original and adversarial explanations. Continuous
features use their training mean. One-hot categorical groups use their most
frequent training category, so the reference represents a valid category rather
than fractional dummy values. During deletion and insertion, every one-hot group
is changed together and ranked by its largest member attribution. The reference
vector and group definition are saved in each experiment's `fidelity.json` and
provenance record.

---

## Key decisions, summarised

| Decision | Rationale |
|---|---|
| Binary labels | Study is attack-vs-normal detection; robustness of the *explanation*, not multiclass accuracy |
| Drop identifiers / Destination Port | Force the model to learn behaviour, not host/service identity (anti-leakage) |
| Re-split both datasets ourselves | Identical, comparable split logic across datasets |
| Scaler fit on train only | No test-statistic leakage |
| Balance train only, freeze test | Realistic, untouched test set for honest evaluation |
| Frozen attack subset | Tractable, reproducible input for the expensive tamper strand |
| ε in normalised space | Comparable perturbation budget across features (within a dataset) |

## Not done (and why)

- **No aggressive feature selection** beyond dropping leaky columns — the study
  evaluates explanations over the full feature set; pruning features would change
  what SHAP/LIME can attribute.
- **No feature-validity constraints** on perturbations — perturbations are
  treated as controlled experimental stimuli, not physically realisable flows
  (stated as a limitation).
- **Global dedup/NaN-drop before split** — acceptable (per-row, deterministic,
  no learned parameters), and standard practice.

## Reproducibility

Given a fixed `run.seed`, the split, the balancing, and the attack subset are
deterministic — the same config reproduces the same frozen `DataBundle`, which is
what the persisted `provenance.json` records.
