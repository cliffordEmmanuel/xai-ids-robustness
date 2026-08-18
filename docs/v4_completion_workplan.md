# V4 Completion Workplan: bounded final evidence

## Objective

Produce a clean, defensible final evidence record without trying to force
KernelSHAP-on-MLP into a result it has not earned. The two completed numerical
calibrations show that its approximate attributions are not stable enough under
the predeclared release rule. That is a measured limitation, not a cue for
unbounded parameter tuning.

## Claim boundary frozen before the final rerun

| Evidence | Final claim allowed | Claim excluded |
|---|---|---|
| TreeSHAP on XGBoost | Robustness and tamper findings for the tree-model/TreeSHAP setup. | That the result applies to all SHAP implementations. |
| LIME on XGBoost and MLP | Primary explainer-held-constant comparison of model families, interpreted with the LIME noise-floor evidence. | A claim that LIME is deterministic or universally representative of all explainers. |
| KernelSHAP on XGBoost | Backend-sensitivity control: whether the XGBoost pattern is reproduced using a model-agnostic SHAP approximation. | A pure architecture comparison. |
| KernelSHAP on MLP | A calibrated numerical non-evaluability finding. | Robustness, tamper-success, or architecture-effect claims. |

The archive directories `results/kernelshap-calibration-v1/` and
`results/kernelshap-calibration-v2/` are the evidence for the last row; they
must not be mixed into `results/v4/`.

## Phase 10: freeze the final configuration set

Create distinct v4 specifications rather than altering prior v3/v4 candidates.

1. **Core fragility grid — 108 rows.**
   - Datasets: CICIDS2017, UNSW-NB15.
   - Valid explainer/model pairs: TreeSHAP/XGBoost, LIME/XGBoost, LIME/MLP.
   - Attacks: FGSM and PGD; epsilon 0.01, 0.03, 0.05; seeds 42, 7, 123.
2. **Core tamper grid — 18 rows.**
   - Same datasets, pairs, and seeds.
   - PGD, epsilon 0.03, top-1 suppression objective.
3. **KernelSHAP XGBoost backend control — 12 rows.**
   - Both datasets, both strands, PGD, epsilon 0.03, seeds 42, 7, 123.
   - Fixed prevalidated candidate: `shap_kernel_nsamples: 1000`,
     `shap_kernel_l1_reg: 0.0001`.
   - Three fresh stability repeats and `require_shap_kernel_stability: true`.
4. **Rank-margin sensitivity — 18 rows, only after core validation.**
   - Same valid three explainer/model pairs, two datasets, and three seeds.
   - PGD, epsilon 0.03; differs only by `target_objective: rank_margin`.

The full final real-data release will therefore contain **156 rows**
(`108 + 18 + 12 + 18`). The core finish line is **138 rows** before the
rank-margin sensitivity.

**Acceptance gate:** every generated config validates; the final specs contain
no KernelSHAP-on-MLP cell; the backend control uses the selected v2 setting.

## Phase 11: smoke gates

Run small, separate real-data smokes before the full grids.

1. A non-KernelSHAP smoke covering both strands and the valid LIME/TreeSHAP
   paths.
2. A strict KernelSHAP-on-XGBoost smoke covering both datasets and both strands
   at `nsamples=1000`, `l1_reg=0.0001`.

The strict backend smoke must validate with zero convergence warnings, finite
attributions, local accuracy within its tolerance, and repeat drift within its
predeclared limit. A failure removes the backend-control claim; it does not
block the LIME/TreeSHAP core release.

**Acceptance gate:** every smoke passes `validate-results` against its exact
specification and its per-flow/output artifacts reconcile with the summaries.

## Phase 12: final core evidence

1. Run the 108-row fragility grid and validate its exact coverage.
2. Run the 18-row tamper grid and validate its exact coverage.
3. Run the 12-row strict KernelSHAP-on-XGBoost control and validate it.
4. Validate the combined 138-row `results/v4/` table before analysis.

**Acceptance gate:** all planned core rows are present exactly once; all run
manifests, per-flow records, query counts, model-quality fields, LIME noise
artifacts, and KernelSHAP control diagnostics validate.

## Phase 13: rank-margin sensitivity

Only after Phase 12 validates, run the focused 18-row rank-margin grid and
revalidate the combined 156-row table. Report it as a sensitivity analysis, not
as a replacement primary outcome.

**Acceptance gate:** top-*k* displacement remains the primary success measure;
rank-margin changes only the attack objective and creates separate experiment
IDs.

## Phase 14: exact-Shapley synthetic validation (time-boxed enhancement)

This is the highest-value optional study, but must remain small and separate
from the IDS evidence release.

1. Use a synthetic binary classification problem with at most eight input
   features, allowing exact Shapley enumeration.
2. Apply a bounded, prediction-preserving explanation attack to a fixed cohort.
3. Compare exact Shapley attributions with the approximate KernelSHAP estimate
   under the same background/reference distribution.
4. Report attribution error, rank agreement, attack-outcome agreement, and
   fresh-seed KernelSHAP variability.

The study validates the explainer/attack/metric pathway in a controlled setting
and can illustrate an approximation-failure mechanism. It must not claim to
reproduce the full real IDS pipeline or be merged into `results/v4/`.

**Time box:** one data generator, one classifier family, one attack setting, and
one concise figure/table. Stop after the predetermined artifacts are produced.

## Phase 15: analysis, thesis integration, and release

1. Run `analyze-results results/v4` only after the combined table validates.
2. Report each dataset/model/explainer condition separately; treat seeds as
   repeatability checks, not independent pooled observations.
3. Use bootstrap intervals only as conditional within-cell precision.
4. For LIME, report raw attack drift alongside benign repeat drift and the
   paired attack-minus-noise result.
5. Report TreeSHAP-versus-KernelSHAP-on-XGBoost as a backend-sensitivity test.
   It is not an architecture comparison.
6. Include both KernelSHAP calibration summaries as a measured limitation; do
   not report their attack outcomes as robustness results.
7. Write a v4 manifest and artifact archive after results, analyses, specs, and
   checksums are frozen.

## Stop rules

- Do not run more production KernelSHAP-on-MLP cells.
- Do not relax the numerical stability threshold to make a candidate pass.
- Do not add models, datasets, explainers, or attack families to the final grid.
- Do not increase seed count beyond 42, 7, and 123 unless a supervisor requires
  it explicitly.
- Do not begin the synthetic study before the Phase 12 core release is running
  or complete.
- If time is constrained, stop after the 138-row core release and document the
  omitted rank-margin/synthetic enhancements as future work.

## Decision gates and finish line

| Gate | Decision | Result if it fails |
|---|---|---|
| G1 | Final specs pass review and generate expected row counts. | Fix configuration only; do not run experiments. |
| G2 | Both smokes validate. | Fix the affected path; core non-KernelSHAP evidence may proceed independently. |
| G3 | 138-row core table validates. | Repair only reconciliation/provenance issues before analysis. |
| G4 | Rank-margin sensitivity validates. | Omit it from the final release; preserve primary results. |
| G5 | Synthetic study completes within its time box. | Treat it as future work, not a blocker. |

The thesis finish line is G3: a validated, reproducible 138-row core evidence
record with bounded claims. G4 and G5 strengthen it but do not justify delaying
submission indefinitely.
