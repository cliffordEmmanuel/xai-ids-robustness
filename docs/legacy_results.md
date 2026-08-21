# Legacy Results Record

The outputs located under `results/v1/` are the pre-improvement
experimental record. They are retained for auditability and must not be mixed
with revised Phase 0+ results.

## Record

- Code revision recorded: `b81ee43d1aa00bd8247e592418208c4f80f741a5`.
- Raw table: `results/v1/results.csv`.
- Observed rows: 169 total: 144 fragility and 25 tamper.
- Intended grid coverage: 144 fragility and 24 tamper rows.
- Legacy/stale row: `unsw-xgboost-shap-tamp-e0.03-s42`.

The stale row uses an old experiment identifier (it omits the attack method),
has incomplete query fields, and is explicitly excluded by `analysis.ipynb`.
It must not enter a final analysis dataset.

## Supersession rationale

These results predate the corrections documented in
[improvement.md](improvement.md): train-only UNSW categorical processing,
shared fidelity baselines, explicit analysis cohorts, repeated LIME noise-floor
measurements, exact per-flow tamper targets, and an explicit explanation-oracle
threat model.

The completed Phase 0 verification run is retained in `results/v2/`. It has
complete grid coverage and schema version 2.0, but it predates the subsequent
methodological implementation phases. Phase 1+ executions write to `results/v3/`
so they cannot overwrite it.

Use these validation commands for the corresponding records:

```bash
uv run xai-ids validate-results results/v2 \
  --spec configs/gridA.spec.yaml \
  --spec configs/gridB.spec.yaml

uv run xai-ids validate-results results/v3 \
  --spec configs/gridA.spec.yaml \
  --spec configs/gridB.spec.yaml
```
