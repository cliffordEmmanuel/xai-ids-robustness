# Why SHAP and LIME

Rationale for choosing SHAP and LIME as the post-hoc explainers evaluated in this
research.

## The two methods

- **LIME** — *Local Interpretable Model-agnostic Explanations* (Ribeiro, Singh &
  Guestrin, KDD 2016). Explains one prediction by sampling perturbed points around
  the instance, weighting them by proximity, and fitting a sparse **linear
  surrogate** to that local neighbourhood. The surrogate's coefficients are the
  feature attributions. It is stochastic (random sampling) and model-agnostic.
- **SHAP** — *SHapley Additive exPlanations* (Lundberg & Lee, NeurIPS 2017).
  Assigns each feature a contribution using **Shapley values** from cooperative
  game theory — the unique attribution satisfying local accuracy, missingness, and
  consistency. Computed exactly for tree ensembles (TreeSHAP) and approximately for
  other models (Kernel SHAP, and network-specific variants).

## Why these two for this study

**1. They are what practitioners actually deploy.** SHAP and LIME are the dominant
post-hoc feature-attribution explainers in ML-based IDS — the tools analysts rely
on to understand why a flow was flagged. Studying *their* robustness speaks to a
real trust problem, not a hypothetical one.

**2. They are model-agnostic — which the design requires.** The central question
compares XGBoost vs MLP. To isolate the *architecture* effect, the explainer must
apply to both models through the same interface. SHAP and LIME treat the model as
a black box, so they do. Gradient-based explainers (Integrated Gradients, Grad-CAM)
are specific to differentiable networks and cannot run on XGBoost, which would make
the cross-architecture comparison impossible.

**3. They use fundamentally different mechanisms — so comparing them is
meaningful.** SHAP is game-theoretic and additive with theoretical guarantees; LIME
is a stochastic local linear surrogate. Because they construct explanations so
differently, they plausibly have different robustness and tamper profiles, making
"SHAP vs LIME" a genuine axis of investigation rather than two variants of the same
thing. Both output the same object — a per-feature attribution vector — so the
stability, fidelity, and drift metrics apply uniformly and enable a clean head-to-
head.

**4. They are the established targets in the adversarial-XAI literature.** The
foundational manipulation work is titled *Fooling LIME and SHAP*
\parencite{slackFoolingLIMESHAP2020}; the Baniecki survey centres on them; and
Mia & Pritom test them in cybersecurity. Using the same two keeps results directly
comparable to that body of work, which this study's architecture-aware, cross-
dataset contribution builds on. Both also rely on input perturbation/sampling to
construct explanations — precisely the surface an input-perturbation attacker
exploits — making them the right subjects for a *tamper* study.

## In one line

SHAP and LIME are the deployed, model-agnostic, mechanistically-distinct,
literature-standard explainers whose shared output makes them directly comparable —
exactly the pair needed to ask "does the explanation survive an attacker, and does
that depend on the model?"

## References

- Ribeiro, Singh & Guestrin (2016), *"Why Should I Trust You?": Explaining the
  Predictions of Any Classifier*, KDD. arXiv:1602.04938.
- Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*,
  NeurIPS. arXiv:1705.07874.
