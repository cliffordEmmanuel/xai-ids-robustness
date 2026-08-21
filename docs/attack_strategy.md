# Attack Strategy

**Framework:** XAI-IDS robustness benchmark
**Status:** Design decision record (implemented through improvement Phase 6)
**Scope:** How adversarial perturbations are generated against XGBoost and MLP
intrusion-detection models, and how the two research strands — explanation
*fragility* and explanation *tamper resistance* — are operationalised.

This document records two load-bearing design decisions and their rationale so
that the `Attack` module can be implemented against a settled specification, and
so the dissertation's methodology and limitations chapters have a single source
of truth.

---

## 1. Threat model

The threat model is **black-box** and **input-space**, and treats prediction
preservation differently by strand:

- **Probability-query black-box.** The attacker can query class probabilities,
  from which labels are derived, but has no access to weights, gradients, native
  backend, or training data.
- **Explanation oracle.** For the targeted strand, the attacker may request an
  attribution array from a separate explanation service. The service owns the
  explainer, any training-background sample, and any TreeSHAP native-model
  access; the attacker receives only arrays and a logical request count. This
  makes the explanation access explicit rather than silently treating it as
  ordinary model access.
- **Query budgets.** Each attack is capped at 125,000 individual probability
  vectors and 250 logical explanation-service requests by the published config.
  Backend-internal prediction calls made by SHAP/LIME are not observable at this
  service boundary and are reported as excluded, not folded into the model-query
  count.
- **Input-space.** Perturbations are applied to the (normalised) feature vector.
- **Prediction preservation is strand-specific.** In the **tamper** strand it is
  *enforced during optimisation* — the whole point is that the explanation is
  redirected while the decision is held fixed, so metrics are computed only on the
  preserved cohort. In the **fragility** strand it is *not* a constraint: the
  attack maximises model loss and may flip the decision, and we measure the
  *incidental* explanation drift over all perturbed flows, reporting the retention
  rate (the fraction whose prediction survived) as a separate diagnostic. Gating
  fragility on the preserved cohort would erase data whenever a model is highly
  brittle — e.g. the CICIDS-2017 XGBoost IDS, whose predictions flip on nearly
  every flow even at ε = 0.01, which is itself a fragility finding.

Perturbations are treated as **controlled experimental stimuli** for measuring
explanation behaviour, not as claims of physically realisable network flows.
Feature-validity constraints are deliberately not enforced (see Limitations).

`epsilon` is defined in **normalised feature space**. It therefore means the same
relative budget *within* a dataset but is **not directly comparable across**
datasets with different dimensionality and correlation structure; results are
reported per dataset.

---

## 2. Two strands, one framework

The study runs two complementary strands that share infrastructure but differ in
what the attack optimises.

| Strand | Attack objective | What it measures | Thesis link |
|---|---|---|---|
| **Fragility** (primary breadth) | Model loss (standard FGSM / PGD) | *Incidental* explanation drift over all perturbed flows (retention reported separately) | RQ1–RQ3 |
| **Tamper resistance** (headline claim) | The explanation itself (targeted) | *Deliberate* manipulation of the explanation while the prediction holds | RQ4 |

The fragility strand answers "how much do explanations wobble under standard
adversarial perturbation, and does architecture moderate it?" The tamper strand
answers "can an adversary deliberately redirect the explanation while the verdict
is preserved?" Only the second is tamper resistance in the strict sense; the
first provides breadth, context, and a screen that localises where explanations
are most fragile.

---

## 3. Decision 1 — attacking a non-differentiable model (XGBoost)

### Problem

FGSM and PGD step in the direction of `sign(∇ₓ loss)`. XGBoost is an ensemble of
piecewise-constant trees, so the gradient of its output with respect to the input
is zero almost everywhere and undefined at split boundaries. Canonical FGSM/PGD
are therefore **undefined on XGBoost** — this is not a tuning issue.

### Options considered

- **Surrogate / transfer.** Train a differentiable proxy, attack it, transfer the
  perturbation. Rejected as the primary method: it injects the surrogate's own
  decision geometry between attack and target, which would confound the central
  XGBoost-vs-MLP comparison (a robustness difference could reflect surrogate
  quality, not architecture).
- **Exact tree evasion (Kantchelian et al., MILP).** The gold standard for tree
  ensembles, but expensive and aimed at label-flipping, not our
  prediction-preserving framing. Retained only as an optional validation
  spot-check on a small subset.
- **Decision-based black-box (HopSkipJump, Boundary).** Query-efficient but
  designed to *cross* the decision boundary — the opposite of our
  prediction-preserving constraint.
- **Zeroth-order PGD (SELECTED).** Keep PGD as the single attack formulation for
  both models; obtain the gradient by **SPSA** (simultaneous perturbation
  stochastic approximation) on the predicted probability. XGBoost's probability
  surface is fine-grained even though its label surface is piecewise-constant, so
  finite-difference / SPSA estimates carry real signal.

### Decision

**Use SPSA-estimated PGD as the unified attack on both models.** FGSM is retained
as the one-step variant. The MLP *could* use autograd, but SPSA is applied to
**both** models because (a) the thesis threat model is explicitly black-box, so
querying the target directly is more faithful than white-box autograd, and (b) a
single attack formulation removes the attack-mechanism confound from the
architecture comparison.

This is faithful to the stated "query outputs, no internals" threat model, avoids
the surrogate artifact, and gives the same ε-ball, the same iterative projection,
and the same objective on both models — the cleanest possible basis for RQ3.

---

## 4. Attack formulation

Per attacked sample, PGD runs for `T` steps. At each step the gradient of the
objective w.r.t. the input is estimated by SPSA:

```
explanations_or_queries per sample ≈ T × (2 × q)
```

where `q` is the number of SPSA perturbation vectors averaged per step
(antithetic, so 2 model evaluations each). The perturbation is projected back onto
the ε-ball (L∞ by default) after each step, and PGD is run as a **bounded
perturbation** (fixed step budget, early-stop before a prediction flip) rather
than to convergence, so it explores the ε-ball instead of racing to the boundary.

- **Fragility strand:** objective = model loss. Evaluate on **all** perturbed
  flows (the perturbation may flip the decision); retention is reported as a
  diagnostic.
- **Tamper strand:** objective = an attribution-manipulation loss (below), with
  prediction preservation enforced in-loop via a penalty / projection.

---

## 5. Decision 2 — the targeted (tamper-resistance) attack

### Objective and primary success outcome

The targeted attack optimises the perturbation to move the explanation. The
default is **targeted suppression**: drive down the attribution of the feature
that the *baseline* explanation ranks most important for the true class — i.e.,
"hide the decisive feature from the analyst." This is the concrete, security-
relevant threat for IDS.

Because the target must generalise across two datasets with different feature
sets, the target is defined by a **rule, not a feature name**: for each sample,
target the top-1 baseline-attributed feature for the predicted class.

The primary attack objective is **suppression**: maximise the negative absolute
attribution of that attack-time target. The primary reported outcome is stricter
and deliberately different: success means the original target has a
deterministic attribution rank **greater than `k`** after perturbation. Relative
target-attribution reduction and rank deterioration are supporting outcomes,
not substitutes for displacement. Ties are resolved stably by feature index.

`rank_margin` is an optional sensitivity objective. It maximises the margin
between the target magnitude and the *k*-th strongest non-target magnitude; its
purpose is to test whether the conclusion depends on optimising suppression
rather than displacement directly. It is run under a separate configuration
grid and never replaces the primary suppression result.

### Why query-based on both models

SHAP (KernelSHAP) and LIME are not differentiable w.r.t. the input (LIME is also
stochastic), so even the MLP cannot use autograd on the explanation objective.
Both models therefore use SPSA over the attribution objective — each SPSA query
recomputes a full explanation. This is the expensive strand; scope is bounded
accordingly (Section 7).

### Constraint

Prediction preservation is enforced **during** optimisation (penalty when a step
would flip the label), not by post-hoc filtering, so every reported tamper result
is a genuine "explanation moved while the decision held" case.

---

## 6. Scope of the tamper-resistance claim

**Fragility grid (cheap, full):** CICIDS2017 + UNSW-NB15 × XGBoost + MLP ×
SHAP + LIME × FGSM + PGD × ε-sweep × seeds. This is the standard benchmark and
answers RQ1–RQ3.

**Tamper-resistance study (targeted, bounded): 8 cells.**

```
{CICIDS2017, UNSW-NB15} × {XGBoost, MLP} × {SHAP, LIME}
```

Both models are included deliberately: RQ3/RQ4 make the XGBoost-vs-MLP comparison
a committed research question, so the tamper claim must itself be comparative
across architectures. Per cell: ~300–500 attacked samples, `T ≈ 50` PGD steps,
SPSA `q ≈ 2`, LIME/KernelSHAP internal `n_samples ≈ 1500`, over a modest ε-sweep
and seed set.

Deferred to limitations / future work: full-grid targeted attacks with deep
ε×seed coverage; exact tree attacks beyond an optional spot-check; feature-valid
(physically realisable) perturbations; explanation-*process* attacks (the Slack
et al. scaffolding family); sensitivity to the choice of attribution target.

---

## 7. Methodological safeguards

- **Evaluation cohort by strand.** For **tamper**, report on the prediction-
  preserved cohort (preservation is enforced in-loop, so this is a stable,
  by-construction set) — this isolates explanation manipulation from decision
  change. For **fragility**, report incidental drift over *all* perturbed flows
  and report the retention rate separately as a diagnostic, rather than gating on
  a preserved subset: gating would introduce selection bias and, for a highly
  brittle model, leave no cohort at all. In both cases the retention rate is
  reported alongside the drift metrics rather than silently filtered.
- **Objective / evaluation separation.** The targeted attack optimises one drift
  measure; robustness is *reported* on independent held-out metrics (top-k
  Jaccard, Spearman, fidelity) so the attack cannot "teach to the test."
- **LIME noise distribution.** Recompute LIME from several fresh, independently
  seeded instances on the *unperturbed* input and store per-flow benign drift.
  Report raw attack drift, the benign distribution, and the attack-minus-noise
  distribution; do not subtract and clamp a single noise estimate to zero.
- **Dataset-agnostic metric parameters.** `k` for top-k Jaccard is fixed (or a
  fixed fraction of feature count) so metrics are comparable across the two
  datasets.

---

## 8. Compute model (target hardware: Apple M4 Max, 14–16 cores, 48 GB)

The workload is **CPU-parallel explanation throughput**, not GPU compute, and is
embarrassingly parallel across samples. XGBoost is CPU-only; the MLP is tiny.
No CUDA is required (Apple GPU is Metal); this is not a GPU-bound problem.

```
queries = cells × samples/cell × T × (2 × q)
wall_clock ≈ queries × time_per_explanation / parallel_workers
```

Per-explanation time is the dominant lever:

- **TreeSHAP (XGBoost):** ~ms → those cells are effectively free.
- **KernelSHAP / LIME:** ~0.3 s at `n_samples ≈ 1500` → a cell is ~minutes to
  ~tens of minutes across ~12 workers.

The full 8-cell targeted study is an **overnight (≈3–4 h) job** on the M4 Max.
The two knobs that make it explode are the explainer's internal `n_samples` and
the `T × q` product; keep both modest. Colab Pro does **not** help — its VMs are
CPU-starved (~2 cores) and the rented GPU sits idle; use it only as an optional
overflow worker for extra cells, since each cell is independent.

---

## 9. Interface and architecture implications

- **`Attack` takes a pluggable objective and oracle interfaces.** Model-loss
  (fragility) and attribution-manipulation (tamper) are two objectives behind
  one interface; the gradient source is `spsa`, supplied with probability and
  (when needed) explanation-oracle responses rather than raw components.
- **`Attack` never owns an `Explainer` or `Model`.** The targeted attack invokes
  the explanation oracle inside its optimisation loop; its privileged backend
  state and background sample remain in the runner-owned service.
- **Constrained optimisation.** Prediction preservation is a first-class part of
  the attack loop, not a post-filter.

---

## 10. Limitations (stated, several quantifiable)

- **Cross-model attack strength is estimation-based.** SPSA yields a *noisy*
  gradient estimate, so a robustness gap could partly reflect estimation quality.
  Mitigation: tune `q`; optionally spot-check against exact Kantchelian evasion on
  a small subset to *quantify* the gap rather than merely assert it.
- **Targeted-B breadth is compute-bounded** (8 cells, modest ε/seed depth).
- **Unconstrained perturbations** may be physically implausible in real flows
  (already framed as controlled stimuli).
- **Target-rule and objective dependence.** Tamper results depend on choosing a
  top-1 baseline target and on the suppression objective; the rank-margin
  sensitivity grid tests the objective choice but not alternative target rules.
- **ε not comparable across datasets** (normalised-space definition).

---

## 11. Mapping to thesis objectives and research questions

- **Objective 1 / RQ1–RQ2** → fragility grid (standard FGSM/PGD; stability,
  fidelity, incidental drift).
- **Objective 3 / RQ3** → architecture moderation, answered across both strands.
- **Objective 4 / RQ4** → targeted tamper-resistance study (8 cells), the direct
  measure of tamper resistance.

---

## Research grounding (decision → citation)

Each attack-module decision and the literature that supports it. Bib keys are for
the thesis `references.bib` (verify exact fields against Zotero).

| Decision | Grounding |
|---|---|
| FGSM / PGD, ε-bounded L∞ | `goodfellowExplainingHarnessingAdversarial2015` (FGSM); `madryDeepLearningModels2018` (PGD) |
| SPSA gradient-free estimation | `uesatoAdversarialRiskDangers2018` (SPSA as a *strong* attack; weak attacks mislead); `spallMultivariateStochasticApproximation1992` (the estimator) |
| Trees are non-differentiable → query-based attack | `kantchelianEvasionHardeningTree2016` |
| Black-box, query-only threat model | `chenZOOZerothOrder2017` (ZOO); `ilyasBlackboxAdversarialAttacks2018` (query-efficient black-box) |
| Tamper: perturb input to change explanation while prediction is preserved | `ghorbaniInterpretationNeuralNetworks2019` (fragile interpretation, prediction unchanged); `dombrowskiExplanationsManipulatedGeometry2019` (targeted explanation manipulation); `banieckiAdversarialAttacksDefenses2024` (I-attack taxonomy) |
| SHAP/LIME are manipulable (motivation) | `slackFoolingLIMESHAP2020`; `miaExplainableVulnerableAdversarial2025` (cyber-tabular) |
| Top-1 suppression objective (hide decisive feature) | `ghorbaniInterpretationNeuralNetworks2019` (targeted top-k saliency); `banieckiAdversarialAttacksDefenses2024` |
| Fragility = explanation sensitivity baseline | `alvarezmelisRobustnessInterpretabilityMethods2018` (local-Lipschitz robustness); `ghorbaniInterpretationNeuralNetworks2019` |
| Noise floor / intrinsic explanation variance | `mullerEmpiricalEvaluationRashomon2023`; `ribeiroHowReliableStable2024` |

**Honest caveat:** Ghorbani, Dombrowski, and Alvarez-Melis are on neural
networks / images. This study's contribution is the tabular-IDS, architecture-
aware, SHAP/LIME-specific version — the gap, not a claim these did it.

---

## References

- Goodfellow, Shlens & Szegedy (2015). *Explaining and Harnessing Adversarial
  Examples.* ICLR. arXiv:1412.6572. — FGSM.
- Madry, Makelov, Schmidt, Tsipras & Vladu (2018). *Towards Deep Learning Models
  Resistant to Adversarial Attacks.* ICLR. arXiv:1706.06083. — PGD.
- Uesato, O'Donoghue, Kohli & van den Oord (2018). *Adversarial Risk and the
  Dangers of Evaluating Against Weak Attacks.* ICML. arXiv:1802.05666. — SPSA as
  a strong gradient-free attack; weak attacks overstate robustness.
- Spall (1992). *Multivariate Stochastic Approximation Using a Simultaneous
  Perturbation Gradient Approximation.* IEEE TAC 37(3):332–341. — SPSA estimator.
- Kantchelian, Tygar & Joseph (2016). *Evasion and Hardening of Tree Ensemble
  Classifiers.* ICML. arXiv:1509.07892. — tree ensembles need non-gradient
  attacks; exact MILP evasion (validation reference).
- Chen, Zhang, Sharma, Yi & Hsieh (2017). *ZOO: Zeroth Order Optimization Based
  Black-box Attacks.* AISec. arXiv:1708.03999. — gradient estimation on
  probabilities.
- Chen, Jordan & Wainwright (2020). *HopSkipJumpAttack.* IEEE S&P.
  arXiv:1904.02144. — decision-based alternative (considered, not used).
- Slack, Hilgard, Jia, Singh & Lakkaraju (2020). *Fooling LIME and SHAP.* AIES.
  arXiv:1911.02508. — explanation-process attacks (out of scope; motivation).
- Ghorbani, Abid & Zou (2019). *Interpretation of Neural Networks Is Fragile.*
  AAAI. arXiv:1710.10547. — explanation fragility.
- Müller et al. (2023). *An Empirical Evaluation of the Rashomon Effect in
  Explainable ML.* arXiv:2306.15786. — intrinsic explanation variance (noise
  floor motivation).
