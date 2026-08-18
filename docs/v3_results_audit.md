# v3 Results Audit — Release Candidate Assessment

## Verdict

`results/v3` is structurally complete and reproducible as an **archived evidence
record**, but it must not yet be presented as the final thesis result release.
The validator accepts all planned coverage and the model/query/preservation gates
pass. However, the KernelSHAP implementation emitted 1,563 numerical convergence
warnings during the final run. The affected KernelSHAP conclusions require a
numerical-stability correction and rerun; warnings must not be suppressed.

This assessment is deliberately separated from the v1 and v2 legacy records.
It describes the schema-3.5 v3 evidence only.

## Evidence scope and integrity

The final coverage command completed successfully against all six planned grid
specifications. There are 210 unique experiment IDs rather than 213 because the
three 1,500-sample LIME ablation conditions are identical to existing Grid A
conditions and are therefore idempotently replaced rather than duplicated.

| Evidence set | Specified cells | Unique rows contributed |
|---|---:|---:|
| Broad fragility grid (Grid A) | 144 | 144 |
| Primary tamper grid (Grid B) | 24 | 24 |
| KernelSHAP-on-XGBoost fragility control | 6 | 6 |
| KernelSHAP-on-XGBoost tamper control | 6 | 6 |
| LIME 1,500/3,000/5,000-sample ablation | 9 | 6 |
| Rank-margin tamper sensitivity | 24 | 24 |
| **Total** | **213 specified** | **210 unique** |

The result table uses schema `3.5`. Its 210 rows have matching manifests,
per-flow files, attack/noise artifacts, and checksums. The analysis directory is
current: its manifest references the checksum of the present `results.csv` and
contains 3,912 conditional bootstrap rows and 612 paired LIME
attack-versus-noise rows.

## Quality gates that pass

All 210 rows meet the predeclared comparison-quality rule: attack-class recall
and F1 are both at least 0.80. The 12 dataset/model/seed fits range from 0.8822
to 0.9993 in attack recall and from 0.8888 to 0.9966 in attack F1.

No run exceeds either recorded query budget. The 54 tamper rows have zero final
prediction flips, as required by that strand. The attack safeguard froze 7,869
candidate flow updates across those runs; this is an informative indication that
the preservation constraint was active, not an attack failure or a label flip.

## Convergence-warning finding

The recorded warnings are not LIME warnings. They come exclusively from SHAP's
KernelSHAP backend, whose feature-selection path invokes scikit-learn's
least-angle regression. The captured warning is the active-set-degeneracy warning
from that solver (it recommends changing its numerical tolerance/iteration
configuration). TreeSHAP and LIME have no recorded convergence warnings.

| Explainer backend | Fragility rows / warnings | Tamper rows / warnings | Affected rows |
|---|---:|---:|---:|
| LIME | 78 / 0 | 24 / 0 | 0 |
| TreeSHAP | 36 / 0 | 12 / 0 | 0 |
| KernelSHAP | 42 / 27 | 18 / 1,536 | 36 |

Every KernelSHAP tamper run is affected. The 1,536 warnings make the
MLP+KernelSHAP tamper results and the KernelSHAP-on-XGBoost backend-control
results unsuitable for final quantitative claims. They remain useful diagnostic
evidence, but must be archived as provisional rather than silently treated as
clean observations.

The likely numerical cause is the deliberately inexpensive
`shap_kernel_nsamples: 100` setting combined with KernelSHAP's LARS feature
selection. This is a configuration/implementation issue to resolve by a focused
stability ablation, not by ignoring or filtering warnings after the fact.

## Cohort availability

Fragility's all-attacked and prediction-preserved cohorts answer different
questions. The latter is unavailable whenever every attacked flow changes its
decision. In particular, 18 of 21 CICIDS2017/XGBoost/SHAP fragility rows and 21
of 24 CICIDS2017/XGBoost/LIME rows have no preserved flows. Their all-flow
results describe combined prediction and explanation fragility; they cannot be
called prediction-preserved explanation drift.

All tamper results retain their final predictions by construction, so their
preserved cohort contains all 500 attacked flows per row.

## Main-result audit

The table below reports the primary suppression objective at PGD epsilon 0.03 as
mean ± seed SD across the three repeats. `Success` is the predeclared primary
outcome: the original top-1 attributed feature finishes below rank 10. `Warn`
is the total recorded explanation-time convergence-warning count across those
three runs.

| Dataset | Model | Explainer/backend | Success | Relative reduction | Rank deterioration | Warn |
|---|---|---|---:|---:|---:|---:|
| CICIDS2017 | MLP | LIME | 0.0% ± 0.0 | 2.6% ± 2.1 | 0.34 ± 0.27 | 0 |
| CICIDS2017 | MLP | KernelSHAP | 29.5% ± 3.2 | 49.7% ± 4.3 | 12.93 ± 1.50 | 311 |
| CICIDS2017 | XGBoost | LIME | 8.7% ± 4.0 | 14.1% ± 2.2 | 3.53 ± 0.96 | 0 |
| CICIDS2017 | XGBoost | TreeSHAP | 0.1% ± 0.1 | 0.6% ± 0.2 | 0.03 ± 0.01 | 0 |
| UNSW-NB15 | MLP | LIME | 0.1% ± 0.1 | 2.8% ± 1.2 | 0.61 ± 0.31 | 0 |
| UNSW-NB15 | MLP | KernelSHAP | 6.9% ± 1.5 | 13.6% ± 3.5 | 2.51 ± 0.61 | 348 |
| UNSW-NB15 | XGBoost | LIME | 0.0% ± 0.0 | 0.7% ± 0.3 | 0.00 ± 0.00 | 0 |
| UNSW-NB15 | XGBoost | TreeSHAP | 20.3% ± 9.5 | 53.1% ± 10.9 | 5.58 ± 1.90 | 0 |

TreeSHAP and LIME rows above are numerically clean, but only support descriptive
within-condition statements. Do not pool their 24 seed runs as independent
samples. The three seeds are repeatability checks within four dataset/model
conditions.

## LIME noise-floor control

LIME's convergence status is clean, but a clean solver does not imply an
attack-induced signal. Across the focused CICIDS2017/XGBoost/PGD ablation, mean
cosine attack drift remained below the repeated benign LIME noise floor:

| LIME samples | Attack cosine drift | Benign cosine drift | Attack minus benign |
|---:|---:|---:|---:|
| 1,500 | 0.0146 | 0.2244 | -0.2098 |
| 3,000 | 0.0137 | 0.1886 | -0.1749 |
| 5,000 | 0.0128 | 0.1621 | -0.1493 |

Increasing LIME's sample count reduces the measured benign noise floor, but it
does not reveal a positive cosine attack signal in this cell. The paired
attack-versus-benign sign-flip analyses should therefore be reported directly;
small LIME cosine drift is not evidence that LIME is robust to the attack.

## Claims permitted now

- The v3 pipeline is auditable, meets the declared model-quality gate, respects
  its observable query budgets, and enforces final prediction preservation for
  tamper.
- In the clean TreeSHAP-on-XGBoost primary cells, targeted top-10 displacement
  is condition-dependent: approximately 0.1% on CICIDS2017 and 20.3% on
  UNSW-NB15 under the stated attack and three seed repeats.
- In the focused LIME control cell, cosine attack movement is lower than its
  repeated benign explanation movement; the experiment does not isolate a
  positive attack effect for that metric.

## Claims blocked pending rerun

- Any quantitative MLP-versus-XGBoost architecture comparison using SHAP.
- Any conclusion based on MLP+KernelSHAP tamper outcomes.
- The KernelSHAP-on-XGBoost backend-control conclusion.
- A general ranking such as "LIME is more robust than SHAP". The explainers use
  different algorithms/backends, and LIME's observed movement must be interpreted
  against its own stochastic noise distribution.

## Required corrective release

1. Preserve v3 intact as the archived, warning-bearing release candidate.
2. Add an explicit KernelSHAP regularisation/stability configuration and run a
   small representative ablation. It must predefine a numerical choice, record
   zero convergence warnings, and verify stable attribution outputs before the
   full rerun. Do not globally suppress `ConvergenceWarning`.
3. Create a new v4 evidence directory, rather than mixing corrected KernelSHAP
   results with v3. Rerun the full planned grids under the frozen corrected
   configuration, then validate and analyse v4.
4. Update the thesis tables from the clean v4 release only. Continue to present
   seed SD/range and conditional per-flow intervals as descriptive precision,
   not as a pooled independent-sample test.
