"""PGD attack (FGSM = single step) with SPSA gradient estimation.

One loop serves both strands via a pluggable objective:

* ``objective='loss'``        -> fragility (incidental drift). Runs the bounded
  steps; prediction preservation is checked afterward (retention meaningful).
* ``objective='attribution'`` -> tamper (targeted). Enforces prediction
  preservation in-loop (revert-on-flip), and only still-active samples are
  perturbed further, so the (expensive) explainer is not re-run on frozen rows.

Perturbations are projected onto the L-inf epsilon-ball in normalised space.

The attack receives a probability oracle and, for tamper, an explanation oracle.
It never receives a raw model, native backend, explainer, or training background.
Query accounting comes from those service boundaries; backend-internal explainer
model calls are explicitly outside the observable query count.
"""

from __future__ import annotations

import numpy as np

from ..interfaces.attack import Attack, AttackResult
from ..oracles import ExplanationOracle, ProbabilityOracle
from ..registry import ATTACKS
from .spsa import spsa_gradient


@ATTACKS.register("pgd")
@ATTACKS.register("fgsm")
class PGDAttack(Attack):
    name = "pgd"

    def __init__(
        self,
        epsilon: float = 0.03,
        steps: int = 50,
        step_size: float = 0.005,
        spsa_q: int = 2,
        spsa_delta: float = 0.01,
        target_rule: str = "top1_suppress",
        target_objective: str = "suppression",
        target_topk: int = 10,
        enforce_prediction_preserved: bool = True,
        seed: int = 42,
    ):
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.spsa_q = spsa_q
        self.spsa_delta = spsa_delta
        self.target_rule = target_rule
        self.target_objective = target_objective
        self.target_topk = target_topk
        self.enforce_prediction_preserved = enforce_prediction_preserved
        self.seed = seed

    def generate(
        self,
        model_oracle: ProbabilityOracle,
        X,
        *,
        objective="loss",
        gradient_source="spsa",
        explanation_oracle: ExplanationOracle | None = None,
    ):
        if gradient_source != "spsa":
            raise NotImplementedError(
                "only gradient_source='spsa' is implemented (autograd path deferred)"
            )
        if not isinstance(model_oracle, ProbabilityOracle):
            raise TypeError("attacks require a ProbabilityOracle, not a raw model")
        if explanation_oracle is not None and not isinstance(explanation_oracle, ExplanationOracle):
            raise TypeError("attacks require an ExplanationOracle, not a raw explainer")
        X0 = np.asarray(X, dtype=float)
        n = len(X0)
        rng = np.random.default_rng(self.seed)
        y_ref = model_oracle.predict(X0)

        if objective == "attribution":
            if explanation_oracle is None:
                raise ValueError("objective='attribution' requires an explanation_oracle")
            base_attr = explanation_oracle.explain(X0)
            if self.target_objective == "rank_margin" and self.target_topk >= base_attr.shape[1]:
                raise ValueError("target_topk must be smaller than the feature count")
            targets = np.argsort(-np.abs(base_attr), axis=1)[:, 0]
        else:
            targets = None

        def objective_fn(Z, idx):
            """Scalar objective to maximise, over rows ``idx`` of the batch."""
            if objective == "attribution":
                if explanation_oracle is None:  # guarded before optimisation; narrows type
                    raise RuntimeError("missing explanation oracle")
                attr = explanation_oracle.explain(Z)
                target_abs = np.abs(attr[np.arange(len(Z)), targets[idx]])
                if self.target_objective == "suppression":
                    return -target_abs
                if self.target_objective == "rank_margin":
                    magnitudes = np.abs(attr).copy()
                    magnitudes[np.arange(len(Z)), targets[idx]] = -np.inf
                    kth_competitor = np.partition(magnitudes, -self.target_topk, axis=1)[
                        :, -self.target_topk
                    ]
                    return kth_competitor - target_abs
                raise ValueError(f"unknown target objective: {self.target_objective}")
            other = 1 - y_ref[idx]
            return model_oracle.predict_proba(Z)[np.arange(len(Z)), other]

        x_adv = X0.copy()
        active = np.ones(n, dtype=bool)
        # A tamper row becomes inactive the first time a candidate perturbation
        # would change its decision.  Keep this separately from the final
        # preservation mask: the latter says what happened, while this records
        # the attack safeguard that caused optimisation to stop for the row.
        frozen_due_prediction_flip = np.zeros(n, dtype=bool)
        for _ in range(self.steps):
            idx = np.where(active)[0]
            if len(idx) == 0:
                break
            Xa = x_adv[idx]
            grad, _ = spsa_gradient(
                lambda Z, idx=idx: objective_fn(Z, idx),
                Xa,
                q=self.spsa_q,
                delta=self.spsa_delta,
                rng=rng,
            )
            cand = np.clip(
                Xa + self.step_size * np.sign(grad),
                X0[idx] - self.epsilon,
                X0[idx] + self.epsilon,
            )
            if self.enforce_prediction_preserved:
                ok = model_oracle.predict(cand) == y_ref[idx]
                x_adv[idx[ok]] = cand[ok]
                active[idx[~ok]] = False  # freeze samples that would flip
                frozen_due_prediction_flip[idx[~ok]] = True
            else:
                x_adv[idx] = cand

        y_adv = model_oracle.predict(x_adv)
        preserved = y_adv == y_ref
        meta = {
            "objective": objective,
            "target_objective": self.target_objective,
            "target_topk": self.target_topk,
            "targets": None if targets is None else targets.tolist(),
            "objective_baseline_attributions": None if targets is None else base_attr,
            "reference_predictions": y_ref.tolist(),
            "adversarial_predictions": y_adv.tolist(),
            "frozen_due_prediction_flip": frozen_due_prediction_flip.tolist(),
            "n_frozen_due_prediction_flip": int(frozen_due_prediction_flip.sum()),
            # Count perturbed *flows*, not changed feature coordinates. This
            # must reconcile with per_flow.csv's n_changed_features > 0 rule.
            "n_perturbed": int(
                (~np.isclose(x_adv - X0, 0.0, atol=1e-12)).any(axis=1).sum()
            ),
            "model_queries": model_oracle.queries,
            "explanation_queries": (
                0 if explanation_oracle is None else explanation_oracle.queries
            ),
            "model_oracle": model_oracle.record(),
            "explanation_oracle": (
                None if explanation_oracle is None else explanation_oracle.record()
            ),
        }
        return AttackResult(
            X_adv=x_adv,
            prediction_preserved=preserved,
            n_queries=model_oracle.queries,
            meta=meta,
        )
