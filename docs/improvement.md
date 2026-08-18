# Improvement Plan: Stronger, Re-runnable XAI Robustness Experiments

## Purpose

This plan strengthens the experimental pipeline before producing replacement
results. It preserves the present outputs as a legacy record, corrects the
identified validity and measurement issues, and narrows the final claims to
what the improved design can support.

## Decisions adopted

| Area | Decision |
|---|---|
| Threat model | Probability-query black box **with an explanation oracle**. The attacker may request explanations, but does not access model weights or gradients. |
| Fragility analysis | Report both all-attacked flows and the prediction-preserved subset. Only the latter may be described as prediction-preserving. |
| LIME instability | Estimate repeated benign explanation drift and report raw attack drift, noise distribution, and uncertainty-adjusted effect. Do not subtract and clamp to zero. |
| Tamper success | Primary outcome: the original top-1 feature leaves the top-*k* set. Supporting outcomes: target attribution reduction and target-rank change. |
| SHAP architecture control | Run KernelSHAP on XGBoost at PGD, epsilon 0.03, for both datasets and three seeds. |
| Result release | Commit a compact, versioned result manifest; archive large artifacts separately with checksums and provenance. |

## Phase 0: preserve and version the current record

The current outputs must remain auditable but must not be mixed with revised
results.

1. Add a results schema version to every row and artifact manifest.
2. Create a legacy manifest that records the current code revision, grid
   configurations, known stale tamper row, and the reasons results are
   superseded.
3. Add `xai-ids validate-results` to check expected grid coverage, unique IDs,
   required fields, matching artifacts, and non-null query fields.
4. Write each revised experimental generation to a new versioned directory
   rather than overwriting an archived record. `results/v1/` is legacy,
   `results/v2/` is the Phase 0 verification run, and Phase 1 starts at
   `results/v3/`.

**Acceptance criteria:** legacy outputs are unchanged; a validator rejects the
legacy stale row from a final dataset; the revised result set has a manifest and
checksums.

## Phase 1: remove UNSW-NB15 preprocessing leakage

UNSW category grouping and encoding must be learned from the training split only.

1. Keep raw cleaning separate from feature encoding.
2. Split the cleaned raw rows before category frequency calculations.
3. Implement a fitted categorical transformer that stores retained categories,
   the `other` bucket, output column order, and an unknown-category policy.
4. Fit that transformer on training rows only, then transform train and test
   rows with the same fitted object.
5. Persist it beside the scaler and include its state in the preprocessing/model
   fingerprint and provenance.

**Acceptance criteria:** a category occurring only in test data cannot alter
training feature columns or category retention; saved and reloaded preprocessing
produces identical arrays.

## Phase 2: make cohorts and per-flow outputs explicit

The pipeline must distinguish model-decision fragility from explanation change
with an unchanged decision.

1. Remove the unused `fixed_cohort` setting. Cohort membership is defined
   independently within each experiment from the recorded final predictions.
2. For fragility, calculate and save metrics for:
   - all attacked flows;
   - prediction-preserved flows, if non-empty.
3. For tamper, retain the enforced prediction-preserved cohort and report any
   rows frozen because a candidate step would flip their prediction.
4. Export per-flow records containing original and adversarial predictions,
   preservation status, L-infinity/L1/L2 perturbation norms, changed-feature
   count, target feature, and attack outcome fields.
5. Add unambiguous, wide result columns such as `n_total`, `n_preserved`,
   `n_flipped`, `all_*`, and `preserved_*`; one result row remains one
   experiment.

**Acceptance criteria:** no result labelled prediction-preserved contains a
flipped prediction; per-flow evidence reconciles with all cohort counts and
attack outcomes, while the checked `attack.npz` attribution artifact allows
all-flow and preserved-only explanation metrics to be regenerated.

## Phase 3: implement the declared explanation-oracle threat model

**Implementation status:** completed in the current worktree. Revised runs use
schema 3.1 and record the declared oracle access, budgets, separate actual query
counts, and the explicit exclusion of backend-internal explainer model queries.

Targeted optimisation needs access to explanations. This must be explicit and
technically separated from direct model access.

1. Add a `threat_model` config section with `model_access`,
   `explanation_access`, and query budgets.
2. Introduce an explanation-oracle interface exposing `explain(X)` and query
   accounting.
3. Route the attack through probability and explanation oracles instead of
   directly consuming raw model/explainer internals.
4. Ensure pure probability-oracle mode prohibits `model.native` and direct
   training-background access from attack code.
5. Record model queries, explanation queries, and a documented estimate or
   explicit exclusion of internal explainer model calls.

**Acceptance criteria:** the configured threat model is recorded in every row;
tests demonstrate that a pure black-box attack cannot access native model state.

## Phase 4: correct deletion and insertion fidelity

Baseline and adversarial explanations must be evaluated against exactly the same
reference input.

1. Derive one per-experiment reference baseline from training data.
2. Pass that fixed baseline to both original and adversarial deletion/insertion
   computations.
3. Attach feature-group metadata for one-hot categoricals.
4. Use valid categorical reference states rather than fractional one-hot means,
   or clearly limit the metric to continuous features where this cannot yet be
   implemented.
5. Save the baseline and feature-group definition in artifacts and provenance.

**Acceptance criteria:** tests prove both fidelity paths receive the same
reference baseline; categorical perturbation/reference states are valid.

## Phase 5: rebuild LIME noise-floor measurement

**Implementation status:** completed in the current worktree. Schema 3.2 writes
`noise_floor.npz` for every run (with per-flow LIME benign-drift samples where
applicable), and `configs/lime_noise_ablation.spec.yaml` defines the focused
1,500/3,000/5,000-sample control study.

The reported `_p025`/`_p975` fields are empirical 2.5th/97.5th percentiles over
paired flow-by-repeat values, not confidence intervals for the grand mean.
These evaluation-time repeats are stored separately and are not charged against
the adversary's configured query budget.

LIME's stochasticity must be measured as a distribution rather than a single
additional explanation call.

1. Add `noise_floor_repeats` and `noise_floor_seeds` to metrics configuration.
2. Create fresh, independently seeded LIME realizations for each benign repeat.
3. Store per-flow benign drift samples and their summary distribution.
4. Report raw attack drift, benign noise-floor mean/interval, and the
   attack-minus-noise effect with uncertainty.
5. Do not clamp adjusted drift to zero or describe indistinguishable drift as
   evidence of robustness.
6. Run a targeted LIME sampling ablation at 1,500, 3,000, and 5,000 samples,
   with repeated fresh seeds, on representative cells.

**Acceptance criteria:** a result can distinguish attack-induced drift from
intrinsic LIME variance; repeated LIME calls use independent random states.

## Phase 6: align the tamper objective and success measure

**Implementation status:** completed in the current worktree. Schema 3.3
stores exact attack-time target attributions in `attack.npz` and reconciles them
with per-flow rank/magnitude outcomes in `per_flow.csv`. The primary summary is
`tamper_success_topk_displacement`; the rank-margin sensitivity grid is defined
in `configs/tamper_rank_margin.spec.yaml`.

The attack and the operational success claim must be traceable at per-flow level.

1. Preserve top-1 suppression as the initial attack objective.
2. Store the exact target selected from the exact baseline realization used by
   the attack.
3. Measure target attribution magnitude before/after, rank before/after, and
   top-*k* membership before/after.
4. Set primary success to `target_rank > k`; report relative attribution
   reduction and rank deterioration as supporting outcomes.
5. Add an optional rank-margin objective that directly optimises displacement
   beyond top-*k* for a sensitivity comparison.

For reproducibility, equal attribution magnitudes use a stable feature-index
tie-break when computing ranks. A success is therefore unambiguously recorded
only when its resulting rank is strictly greater than `target_topk`.

**Acceptance criteria:** per-flow targets and success summaries agree exactly;
all recorded successes satisfy the configured rule.

## Phase 7: comparability controls and analysis inputs

**Implementation status:** completed in the current worktree. Schema 3.5
preserves frozen-test model performance, class counts, a transparent
comparison-eligibility flag, explainer backend/sampling identity, and adds
explanation-time convergence-warning counts. The backend-control grids are
`configs/shap_kernel_xgboost_fragility.spec.yaml` and
`configs/shap_kernel_xgboost_tamper.spec.yaml`.

**Model-quality policy:** comparative explanation claims require both attack-class
recall and attack-class F1 to be at least 0.80 for the relevant dataset/model/seed.
This is a reporting guard, not an execution filter: sub-threshold models are still
stored, validated, and discussed as model-quality failures rather than discarded.

1. Add accuracy, precision, recall, F1, and relevant class counts to the master
   result table for each dataset/model/seed.
2. Define and document any model-quality threshold required for explanatory
   comparisons.
3. Run the focused SHAP backend control:
   - KernelSHAP on XGBoost;
   - PGD, epsilon 0.03;
   - CICIDS2017 and UNSW-NB15;
   - seeds 42, 7, and 123;
   - fragility first, followed by tamper if the compute budget permits.
4. Treat LIME as the primary explainer-held-constant comparison between model
   families. Describe SHAP results as architecture-plus-backend unless the
   control supports a narrower conclusion.

**Acceptance criteria:** the analysis can present model performance beside each
robustness result and identify backend-controlled SHAP comparisons.

## Phase 8: test, validate, and rerun

**Implementation status:** smoke and analysis tooling are completed in the
current worktree. `configs/phase8_smoke.spec.yaml` exercises both strands and
both explainers on a small real-data workload. `xai-ids analyze-results` writes
conditional per-cell bootstrap intervals and LIME paired attack-versus-benign
noise summaries only after the input result directory validates.

The runner records `n_explainer_convergence_warnings` and the distinct messages
in each run's `metrics.json`. Do not suppress LIME/LARS convergence warnings:
inspect their count in the smoke outputs and either resolve the numerical issue
or make a documented exclusion/limitation decision before final execution.

1. Add unit and integration tests for all preceding acceptance criteria.
2. Add a small synthetic end-to-end grid covering both strands and explainers.
3. Run static checks, the full test suite, and result validation before any
   expensive grid execution.
4. Run a smoke grid on real data, inspect per-flow exports and query accounting,
   then freeze code/configuration versions. Provenance records the git revision,
   whether the working tree was dirty, the full config, and dependency versions.
5. Run the primary grid:
   - fragility: existing breadth, with all-flow and preserved-flow reporting;
   - tamper: eight cells by three seeds, repeated LIME noise estimates;
   - backend control: the focused KernelSHAP-on-XGBoost study above.
6. Produce a compact tracked final manifest plus an archive containing large
   artifacts, configurations, provenance, and checksums.

**Acceptance criteria:** every final CSV row validates, has matching artifacts,
can be traced to configuration and dependency versions, and can be regenerated
from its released inputs.

## Phase 9: stabilise KernelSHAP before releasing v4

**Implementation status:** completed in the current worktree. Schema 3.6 records
explicit KernelSHAP L1 regularisation, convergence-warning counts, finite-value
checks, additivity residuals, and fresh-seed repeat drift. The strict result
validator is a release guard rather than a warning suppressor. The calibration
and selection procedure is documented in
[`kernelshap_stability.md`](kernelshap_stability.md).

The v3 audit found 1,563 KernelSHAP least-angle-regression convergence warnings,
so no final KernelSHAP thesis conclusion may rely on that evidence record.

1. Calibrate explicit numerical settings across representative datasets, model
   families, strands, and a small sample of true-positive flows.
2. Select a single setting using only warnings, finiteness, additivity, and
   repeatability; never select by attack outcome.
3. Require the selected setting to pass a strict eight-cell real-data smoke
   grid before final execution.
4. Use schema 3.6 and strict KernelSHAP validation for all new v4 evidence.

**Phase 9b status:** configured after `kernelshap-calibration-v1` produced no
candidate passing every numerical check. The distinct v2 calibration retains
the predeclared gates and tests larger coalition-sample budgets (1,000/3,000)
with explicit L1 settings (0.0001/0.001). Do not launch the Phase 9 strict smoke
until its candidate summary contains an all-cell pass.

## Final completion scope

The calibration outcome changes the final-evidence plan: do not launch a
production KernelSHAP-on-MLP grid. The bounded v4 completion plan, row counts,
claim boundaries, release gates, and time-boxed synthetic enhancement are in
[`v4_completion_workplan.md`](v4_completion_workplan.md).

**Acceptance criteria:** every final KernelSHAP row has zero convergence
warnings, finite attributions, acceptable local accuracy, and stable fresh-seed
repeats; the saved diagnostic artifacts reconcile exactly with `results.csv`.

## Analysis and claims after rerun

- Treat four dataset/model combinations as experimental conditions and three
  seeds as repeatability checks. Do not apply pooled independent-sample tests to
  the 12 runs per explainer.
- Use per-flow bootstrap intervals only as conditional within-cell precision
  estimates: they do not create independent conditions or justify broad
  population claims.
- For LIME, compare attack drift to the repeated benign drift distribution only
  after Phase 5's repeated measurements exist.
- State architecture results as model-family effects in this pipeline. Use LIME
  for the primary explainer-held-constant comparison and use the SHAP control to
  bound the backend confound.

## Implementation order

1. Phase 0 — version outputs.
2. Phase 1 — remove data leakage.
3. Phase 4 — correct fidelity baseline.
4. Phase 2 — cohort/per-flow schema.
5. Phase 5 — LIME repeats.
6. Phase 6 — tamper success alignment.
7. Phase 3 — threat-model oracle boundary.
8. Phase 7 — comparability controls.
9. Phase 8 — tests and reruns.
10. Phase 9 — KernelSHAP calibration and strict v4 release gate.

This order fixes data validity and measurement validity before spending compute
on revised experiments.
