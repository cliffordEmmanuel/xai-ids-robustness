# KernelSHAP numerical-stability calibration and release gate

## Why v4 needs a separate calibration

The archived v3 run used SHAP's implicit `num_features(10)` LARS feature
selection. Its KernelSHAP calls emitted 1,563 least-angle-regression convergence
warnings, including every KernelSHAP tamper cell. The complete audit is in
[v3_results_audit.md](v3_results_audit.md). Those rows remain an auditable record,
but they are not final evidence for KernelSHAP claims.

Schema 3.6 makes the relevant numerical choices explicit and records, for every
KernelSHAP run:

- the exact `l1_reg` setting and number of coalition samples;
- the count and text of all explanation-time convergence warnings;
- whether all attributions are finite;
- the largest local-accuracy (additivity) residual;
- per-flow cosine drift from three fresh, independently seeded KernelSHAP
  realisations using the same background sample.

No warnings are hidden or filtered. The release validator fails a strict final
KernelSHAP row unless every configured numerical check passes.

## Calibration protocol

`configs/kernelshap_stability_calibration.spec.yaml` is a 32-cell, non-evidence
calibration grid: two datasets, two model families, both attack strands, two
coalition-sample budgets (100 and 300), and two explicit L1 settings (`0.000001`
and `0.0001`). It uses 24 true-positive flows and three attack steps only to
exercise the same code paths economically. Its `require_shap_kernel_stability`
setting is intentionally `false`: failures must be captured so candidates can be
ruled out rather than aborting the grid.

Run and validate it as follows:

```bash
uv run xai-ids validate configs/kernelshap_stability_base.yaml
uv run xai-ids gen-grid configs/kernelshap_stability_calibration.spec.yaml
uv run xai-ids grid configs/kernelshap_stability_calibration.spec.yaml
uv run xai-ids validate-results results/kernelshap-calibration-v1 \
  --spec configs/kernelshap_stability_calibration.spec.yaml
uv run xai-ids analyze-results results/kernelshap-calibration-v1
```

The last command creates these selection inputs:

- `analysis/kernel_shap_stability.csv` — one numerical-diagnostic row per
  calibration cell;
- `analysis/kernel_shap_candidate_summary.csv` — one row per
  `(shap_kernel_nsamples, shap_kernel_l1_reg)` candidate.

Choose one candidate only if `all_cells_pass` is true: every representative cell
must have zero convergence warnings, finite attributions, a maximum additivity
error at or below `0.000001`, and maximum fresh-repeat cosine drift at or below
`0.05`. Do not use tamper success, explanation drift, or model quality when
choosing the numerical setting. If no candidate passes, expand this calibration
grid with a larger `shap_kernel_nsamples` or stronger explicit numeric L1 value,
then repeat the calibration.

## Phase 9b follow-up after the v1 calibration

The completed `kernelshap-calibration-v1` run has no eligible candidate. Its
least-bad setting (`nsamples: 300`, `l1_reg: 0.0001`) removed convergence
warnings but still passed only two of eight representative cells because the
fresh-repeat drift was too large. Therefore the strict smoke must not be run
from that result.

`configs/kernelshap_stability_phase9b.spec.yaml` defines a separate, preserved
follow-up: it tests `nsamples: 1000` and `3000`, crossed with `l1_reg: 0.0001`
and `0.001`. It deliberately keeps all other design choices and thresholds
unchanged. `1e-6` is excluded because v1 consistently raised convergence
warnings; `0.01` is excluded because preliminary checks showed a risk of
degenerate all-zero attributions.

```bash
uv run xai-ids validate configs/kernelshap_stability_phase9b_base.yaml
uv run xai-ids gen-grid configs/kernelshap_stability_phase9b.spec.yaml
uv run xai-ids grid configs/kernelshap_stability_phase9b.spec.yaml
uv run xai-ids validate-results results/kernelshap-calibration-v2 \
  --spec configs/kernelshap_stability_phase9b.spec.yaml
uv run xai-ids analyze-results results/kernelshap-calibration-v2
```

Do not alter the `0.000001` additivity or `0.05` repeat-drift gate while
interpreting this run. If no v2 candidate has `all_cells_pass = true`, diagnose
the estimator further before relaxing a gate: in particular, use common random
coalition samples for the baseline-versus-adversarial comparison while retaining
fresh-seed repeats as an external stochasticity check.

## Release procedure after selection

The Phase 9 smoke and production grids currently contain the initial
pre-registered profile `nsamples: 100`, `l1_reg: 0.0001`. Retain it only if the
candidate summary selects it. Otherwise change the two values consistently in:

- `configs/phase9_kernelshap_smoke_base.yaml`;
- the `kernel_shap` section of `configs/gridA.spec.yaml`,
  `configs/gridB.spec.yaml`, `configs/tamper_rank_margin.spec.yaml`,
  `configs/shap_kernel_xgboost_fragility.spec.yaml`, and
  `configs/shap_kernel_xgboost_tamper.spec.yaml`.

Then run the strict release gate. It covers both datasets, both model families,
and both attack strands with the final setting:

```bash
uv run xai-ids grid configs/phase9_kernelshap_smoke.spec.yaml
uv run xai-ids validate-results results/smoke-v4-kernelshap \
  --spec configs/phase9_kernelshap_smoke.spec.yaml
```

Only after that command returns `OK` should the full v4 grids be started. New
results are isolated under `results/v4/`; v1–v3 are never overwritten.

For a strict v4 result, `validate-results` rejects any KernelSHAP row with a
warning, non-finite attribution, additivity failure, insufficient repeats, or
repeat drift beyond its predeclared tolerance. TreeSHAP and LIME rows are marked
`not_applicable` for this check, so the mixed grids retain a stricter evidence
standard for KernelSHAP without treating other explainers as failures.
