# Provisional Thesis Results Summary — v3

## Status

This summary is suitable for drafting the results chapter and discussing the
experimental design. It is **not** final thesis evidence: the accompanying
[v3 audit](v3_results_audit.md) identifies KernelSHAP convergence warnings that
block the affected comparisons until a corrected v4 rerun is complete.

## Design statement

The v3 release evaluates two datasets (CICIDS2017 and UNSW-NB15), two model
families (MLP and XGBoost), and LIME/SHAP explanations under a probability-query
black-box attack with an explicit explanation oracle. Fragility and targeted
tamper are separate strands. Three seeds repeat each dataset/model condition;
they demonstrate within-condition repeatability and are not twelve independent
conditions.

All 210 released rows satisfy the predefined model-quality guard (attack-class
recall and F1 of at least 0.80), and all 54 tamper rows preserve the final model
decision. Per-flow bootstrap intervals are conditional within-cell precision
estimates only.

## Defensible preliminary findings

### Targeted explanation tampering is condition-dependent

For the numerically clean TreeSHAP/XGBoost primary cells at epsilon 0.03, the
mean top-10 displacement rate was 0.1% (SD 0.1%) on CICIDS2017 and 20.3% (SD
9.5%) on UNSW-NB15. The latter coincided with a 53.1% mean target-attribution
reduction and a mean rank deterioration of 5.58. This supports a narrow,
pipeline-specific statement: even with final predictions preserved, the same
bounded targeted attack can have materially different explanation effects across
datasets and model conditions.

The result does not establish a population rate, a universal weakness of
TreeSHAP, or a causal architecture-only effect.

### LIME requires a noise-aware interpretation

The focused CICIDS2017/XGBoost LIME ablation found attack cosine drift below its
repeated benign explanation drift at every sample size tested. Benign cosine
drift declined from 0.2244 at 1,500 samples to 0.1621 at 5,000, but the
attack-minus-benign effect remained negative. Accordingly, wording such as
"LIME was robust because its drift was small" is not supported. The defensible
statement is that the attack signal was not distinguishable from, and was smaller
than, LIME's measured stochastic variation for this cell and metric.

### Prediction-preserved fragility is not available in every cell

For many CICIDS2017/XGBoost fragility runs, all attacked flows changed their
predictions. Those results remain useful for model-plus-explanation fragility,
but they cannot substantiate a claim about explanation movement with an unchanged
decision. The results chapter should show all-flow and preserved-flow metrics
side by side and explicitly mark an empty preserved cohort as unavailable.

## Results that must wait for v4

KernelSHAP produced least-angle-regression convergence warnings in all 18
KernelSHAP tamper rows and some KernelSHAP fragility rows. Therefore do not use
the attractive-looking MLP+KernelSHAP tamper rates, nor the KernelSHAP-on-XGBoost
control, as final evidence. The appropriate response is a pre-specified
numerical-stability correction and clean rerun, not exclusion of warnings from
the report.

## Recommended Results-chapter structure

1. **Experimental integrity:** model-quality table, query-accounting policy,
   per-flow artifact availability, and prediction-preservation cohorts.
2. **Fragility:** present all-flow and preserved-flow results separately. Mark
   unavailable preserved cohorts rather than averaging them away.
3. **LIME stochasticity:** show raw attack drift, benign repeat distribution,
   conditional interval, and paired sign-flip result. Do not report only an
   adjusted/clamped scalar.
4. **Tamper:** make top-10 target displacement the primary outcome; give target
   reduction and rank deterioration as supporting outcomes.
5. **Controls and limitations:** separate TreeSHAP, LIME, and corrected
   KernelSHAP evidence. State that LIME carries the explainer-held-constant
   architecture comparison, while the KernelSHAP control resolves the SHAP
   backend confound only after v4.
