# V4 final-results status and documentation handover

**Status date:** 2026-08-21  
**Frozen code revision:** `97c553b` (`v4-release-candidate`)  
**Raw release directory:** `results/v4/`
**Validated analysis directory:** `results/v4-analysis/` (126 rows)

## Purpose of this record

This document records what was implemented and run for v4, which evidence was
used for thesis analysis, and why the incomplete KernelSHAP control was not
resumed or interpreted as a robustness result. It is the final handover record
for the thesis results, methods, and limitations chapters.

## What has been completed

### Reproducible v4 release preparation

- The v4 code and configuration set was frozen at revision `97c553b`, tagged
  `v4-release-candidate`.
- Separate specifications were prepared for the primary fragility grid, primary
  tamper grid, the KernelSHAP/XGBoost backend control, smoke tests, and the
  optional rank-margin sensitivity.
- Result provenance, per-flow reconciliation, model-quality checks, LIME
  noise-floor outputs, and KernelSHAP numerical-stability diagnostics are part
  of the v4 validation workflow.
- KernelSHAP-on-MLP was deliberately excluded from v4 production experiments:
  the earlier calibrations showed that it did not meet the predeclared
  numerical-stability requirement.

### Calibration and smoke evidence

- KernelSHAP calibration archives were retained separately in
  `results/kernelshap-calibration-v1/` and
  `results/kernelshap-calibration-v2/`. The XGBoost-only candidate used for
  the backend control was `nsamples=1000` with `l1_reg=0.0001`.
- The v4 non-KernelSHAP smoke passed.
- The strict v4 KernelSHAP/XGBoost smoke passed for both datasets, both
  strands, and seeds 42, 7, and 123. Its 12-flow attack cohort checked that
  the end-to-end path executed and its artifacts reconciled; it was not a
  substitute for the 500-flow production stability test.

### Primary v4 evidence

The following pre-specified primary grids are complete:

| Evidence set | Planned rows | Saved rows | Status |
|---|---:|---:|---|
| Fragility: TreeSHAP/XGBoost, LIME/XGBoost, LIME/MLP | 108 | 108 | Complete |
| Tamper: the same valid pairs | 18 | 18 | Complete |
| **Valid primary-core total** | **126** | **126** | Validated and analysed |

The saved primary rows cover CICIDS2017 and UNSW-NB15; FGSM and PGD at
epsilon 0.01, 0.03, and 0.05 for fragility; and PGD at epsilon 0.03 with the
top-1 suppression objective for tamper. Each condition uses seeds 42, 7, and
123 as repeatability runs, not as independent population samples.

## KernelSHAP/XGBoost backend control: stopped and excluded

The optional backend-sensitivity control was designed as 12 strict production
rows (two datasets, two strands, three seeds) at 500 attacked flows per row.
The process was interrupted during the tenth row after **9 of 12** rows had
been written. The completed rows are retained as audit evidence but are not
eligible for robustness or tamper analysis.

| Completed control rows | Result |
|---|---|
| CICIDS2017 fragility, seeds 42/7/123 | All failed the repeat-drift gate |
| CICIDS2017 tamper, seeds 42/7/123 | All failed the repeat-drift gate |
| UNSW-NB15 fragility, seed 42 | Failed the repeat-drift gate |
| UNSW-NB15 fragility, seeds 7/123 | Passed the repeat-drift gate |
| UNSW-NB15 tamper, seeds 42/7/123 | Not run; do not resume |

The failure was not caused by non-finite attributions, additivity failure, or
convergence warnings. All nine completed rows had zero explainer convergence
warnings and negligible maximum additivity error (at most
`4.44e-16`). The failure was repeatability: seven rows had maximum cosine
drift between fresh KernelSHAP explanations above the frozen `0.05` limit
(approximately `0.066` to `0.123`).

This outcome supersedes the small smoke for the substantive backend-control
claim. A 12-flow smoke can establish that a code path and artifact contract
work; it cannot establish stability across a 500-flow production cohort. The
threshold must not be relaxed after seeing the result. Therefore:

- do **not** run the remaining three UNSW-NB15 tamper control configurations;
- do **not** claim that KernelSHAP corroborates the TreeSHAP/XGBoost pattern;
- do retain the partial control output as reproducibility and limitation
  evidence.

## Current artifact accounting

`results/v4/results.csv` currently contains **135 rows**:

| Component | Rows |
|---|---:|
| Valid LIME primary rows | 84 |
| Valid TreeSHAP primary rows | 42 |
| Incomplete KernelSHAP/XGBoost control | 9 |
| **Total raw rows** | **135** |

The raw directory must not be passed unfiltered to final aggregate analysis,
because the nine excluded control rows make a combined strict validation fail.
It is preserved unchanged. The derived, traceable 126-row analysis input is
`results/v4-analysis/`; its full local manifest is
`results/v4-analysis/included_manifest.json`, and its compact tracked summary is
[`v4_analysis_manifest.json`](v4_analysis_manifest.json).

## Claim boundaries for the thesis

The valid v4 primary evidence supports the following bounded claims:

| Evidence | Claim that may be made | Claim that must not be made |
|---|---|---|
| TreeSHAP on XGBoost | Findings for the XGBoost/TreeSHAP implementation and threat model tested. | Findings for all SHAP algorithms or all model families. |
| LIME on XGBoost and MLP | The main model-family comparison while holding this explainer method fixed, reported alongside its benign noise floor. | That LIME is deterministic or representative of all explainers. |
| KernelSHAP on MLP | Numerical non-evaluability under the stated calibration/release rule. | Any robustness, tamper, or architecture-effect result. |
| KernelSHAP on XGBoost | A failed strict backend-control attempt and a measured limitation of the approximate attribution procedure at production scale. | Corroboration of TreeSHAP or a backend-independent SHAP result. |

## Completed analysis and handover

- `results/v4-analysis/` was derived with exactly the 126 primary-core rows.
  Its full manifest includes every included ID, every excluded KernelSHAP
  control row, the raw-result checksum, and the frozen source revision.
- `xai-ids validate-results results/v4-analysis` completed successfully.
  All included rows pass the model-quality guard: attack-class recall and
  $F_1$ are at least 0.80.
- `xai-ids analyze-results results/v4-analysis` produced the per-flow bootstrap,
  LIME attack-versus-noise, and KernelSHAP diagnostic summaries used for the
  thesis tables and figures.
- The thesis distinguishes the 12-flow smoke gate from the 500-flow production
  stability assessment, reports seeds as repeatability checks, and treats the
  KernelSHAP results as numerical-reliability findings rather than attack
  outcomes.

The optional rank-margin sensitivity and synthetic exact-Shapley study remain
future work. They are not required to interpret the completed 126-row primary
release.
