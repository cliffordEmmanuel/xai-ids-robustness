# Explainers — Implementation

How SHAP and LIME are implemented in the framework (Phase 4). For *why* these two
explainers were chosen, see `explainer_choice.md`.

Code: `src/xai_ids/explainers/` (`shap_explainer.py`, `lime_explainer.py`),
interface in `src/xai_ids/interfaces/explainer.py`.

## The contract

Every explainer implements:

```python
explain(model, X, background=None) -> np.ndarray   # (n_samples, n_features)
```

- Output is `(n_samples, n_features)` attributions **for the positive (attack)
  class**, in the normalised feature space.
- Same shape and semantics for SHAP and LIME, so the Phase-6 metrics compare them
  directly.
- `background` is a reference sample (train data) needed by KernelSHAP and LIME;
  TreeSHAP ignores it. Added to the signature because both stochastic explainers
  need reference statistics.

Two shared helpers live on the interface:

- `target_features(attributions, k=1)` — indices of the top-`k` features by
  `|attribution|` per sample; used by the tamper strand's suppression rule.
- `noise_floor(model, X, background, repeats)` — intrinsic attribution variance
  from repeated calls; ~0 for TreeSHAP (deterministic), >0 for KernelSHAP/LIME.

Heavy libraries (`shap`, `lime`) are **lazy-imported inside methods**, so
`xai-ids list explainers` and the core CLI work without the ml extra installed.

## SHAP (`shap_explainer.py`) — per-architecture backend

SHAP dispatches on `backend` (from `explainer.shap_backend` in config):

- **`tree` (XGBoost):** `shap.TreeExplainer(model.native)` — exact and fast, but
  needs the underlying estimator, exposed via the model's `native` property.
- **`kernel` (MLP):** `shap.KernelExplainer(predict_proba_positive, background)` —
  model-agnostic approximation using only `predict_proba` + a background sample.

Key implementation details:

- **Positive-class normalisation.** SHAP's output shape varies by version (list
  `[neg, pos]`, or 3-D `(n, d, classes)`, or `(n, d)`). `_positive_2d` collapses
  all forms to the attack-class `(n, d)` array.
- **Background sampling.** `_sample` takes up to `background_size` rows (seeded) so
  KernelSHAP stays tractable.
- **Engine caching.** The `TreeExplainer`/`KernelExplainer` object is built once
  and cached on `self._explainer`, so the Phase-5 tamper loop (hundreds of calls)
  reuses it instead of rebuilding.

## LIME (`lime_explainer.py`) — one uniform path

LIME is fully model-agnostic, so the *same* code handles XGBoost and MLP. It only
needs `predict_proba`.

- Builds a `LimeTabularExplainer` once (cached) from the background sample.
- Loops over instances, calling `explain_instance(x, model.predict_proba,
  num_features=d, num_samples=…, labels=(1,))` and assembling the attack-class
  weights into `(n, d)`.
- **`discretize_continuous=False`** so weights are per-*raw*-feature — the same
  space as SHAP, so metrics are comparable.
- **Seeded** (`random_state`) for determinism; it is the main consumer of the
  noise-floor helper because its sampling is stochastic.
- It is a per-instance loop (inherent to LIME) — the slow explainer.

## Per-architecture summary

| | XGBoost | MLP | needs |
|---|---|---|---|
| SHAP | TreeSHAP — exact, deterministic | KernelSHAP — approximate, stochastic | tree: `model.native`; kernel: `predict_proba` + background |
| LIME | same (black-box) | same (black-box) | `predict_proba` + background |

## The `explain` command

`xai-ids explain <config>` (config-only):

1. build dataset, `set_seed`;
2. train the model in-memory (the Phase-7 runner will instead load the cached
   model);
3. construct the explainer from config (`_make_explainer`): SHAP gets
   `backend=shap_backend`, LIME gets `num_samples=lime_num_samples`; both get the
   run seed;
4. compute baseline attributions on the frozen attack subset with
   `background = X_train`;
5. save `baseline_attributions.npz` (+ `provenance.json`, `run.log`) to
   `results/<ds>-<model>-<explainer>-s<seed>/`.

These baseline attributions are what the tamper strand diffs against.

## Config mapping

```yaml
explainer:
  name: shap            # {shap, lime}
  shap_backend: tree    # tree (XGBoost) | kernel (MLP); validated: tree ⇒ xgboost
  lime_num_samples: 1500
  cache_baseline: true
```

## Research grounding (decision → citation)

Bib keys are for the thesis `references.bib` (verify exact fields against Zotero).

| Decision | Grounding |
|---|---|
| SHAP (additive Shapley-value attributions) | `lundbergUnifiedApproachInterpreting2017` |
| TreeSHAP — exact attributions for XGBoost | `lundbergLocalExplanationsGlobal2020` |
| KernelSHAP — model-agnostic, for the MLP | `lundbergUnifiedApproachInterpreting2017` |
| LIME (local linear surrogate) | `ribeiroWhyShouldITrust2016` |
| Model-agnostic post-hoc explainers for IDS | `hermosillaExplainableAIForensic2025`; `mohaleEvaluatingMachineLearningbased2025` |
| SHAP/LIME reliability & instability (motivates the robustness study) | `ribeiroHowReliableStable2024`; `bhattExplainableMachineLearning2020`; `kaurInterpretingInterpretabilityUnderstanding2020` |
| Intrinsic explanation variance / noise floor (Rashomon effect) | `mullerEmpiricalEvaluationRashomon2023` |
| Fidelity of explanations varies by method/architecture | `miro-nicolauComprehensiveStudyFidelity2024` |

See `explainer_choice.md` for the full rationale narrative.

## Caveats (for methodology / limitations)

- **xgboost pinned `<3`.** SHAP's TreeExplainer cannot parse xgboost 3.x's model
  format yet (`base_score` became a vector), so the ml extra pins `xgboost<3`
  (currently 2.1.4).
- **SHAP is exact for XGBoost but approximate for the MLP** (TreeSHAP vs
  KernelSHAP). A cross-architecture SHAP difference could partly reflect
  exact-vs-approximate estimation, not architecture — mitigate with a high
  KernelSHAP sample count and by reporting the noise floor.
- **Stochasticity.** TreeSHAP is deterministic; KernelSHAP and LIME are not — hence
  the noise-floor subtraction (Phase 6) applies to them.
- **Cost.** TreeSHAP is ~free; KernelSHAP and LIME make many `predict_proba` calls
  per instance, which is what makes the MLP-SHAP and all-LIME tamper cells the
  expensive ones.
