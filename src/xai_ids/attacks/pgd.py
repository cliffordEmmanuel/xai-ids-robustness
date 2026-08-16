"""PGD attack (FGSM = single step) with SPSA gradient estimation.

One loop serves both strands via a pluggable objective:

* ``objective='loss'``        -> fragility (incidental drift). Runs the bounded
  steps; prediction preservation is checked afterward (retention meaningful).
* ``objective='attribution'`` -> tamper (targeted). Enforces prediction
  preservation in-loop (revert-on-flip); calls the explainer each step.

Perturbations are projected onto the L-inf epsilon-ball in normalised space.
"""

from __future__ import annotations

import logging

import numpy as np

from ..interfaces.attack import Attack, AttackResult
from ..registry import ATTACKS
from .objectives import attribution_objective, loss_objective
from .spsa import spsa_gradient

_log = logging.getLogger("xai_ids")


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
        enforce_prediction_preserved: bool = True,
        seed: int = 42,
    ):
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.spsa_q = spsa_q
        self.spsa_delta = spsa_delta
        self.target_rule = target_rule
        self.enforce_prediction_preserved = enforce_prediction_preserved
        self.seed = seed

    def generate(self, model, X, *, objective="loss", gradient_source="spsa", explainer=None):
        if gradient_source != "spsa":
            raise NotImplementedError(
                "only gradient_source='spsa' is implemented (autograd path deferred)"
            )
        X0 = np.asarray(X, dtype=float)
        n = len(X0)
        rng = np.random.default_rng(self.seed)
        y_ref = model.predict(X0)
        n_queries = 0

        if objective == "attribution":
            if explainer is None:
                raise ValueError("objective='attribution' requires an explainer")
            targets = explainer.target_features(explainer.explain(model, X0), k=1)[:, 0]
            f = attribution_objective(model, explainer, targets)
        else:
            targets = None
            f = loss_objective(model, y_ref)

        x_adv = X0.copy()
        active = np.ones(n, dtype=bool)
        for _ in range(self.steps):
            grad, calls = spsa_gradient(
                f, x_adv, q=self.spsa_q, delta=self.spsa_delta, rng=rng
            )
            n_queries += calls
            cand = np.clip(
                x_adv + self.step_size * np.sign(grad), X0 - self.epsilon, X0 + self.epsilon
            )
            if self.enforce_prediction_preserved:
                ok = model.predict(cand) == y_ref
                n_queries += n
                update = active & ok
                x_adv[update] = cand[update]
                active &= ok  # a sample that would flip is frozen for the rest
                if not active.any():
                    break
            else:
                x_adv = cand

        preserved = model.predict(x_adv) == y_ref
        n_queries += n
        meta = {
            "objective": objective,
            "targets": None if targets is None else targets.tolist(),
            "n_perturbed": int((x_adv != X0).any(axis=1).sum()),
        }
        return AttackResult(
            X_adv=x_adv, prediction_preserved=preserved, n_queries=n_queries, meta=meta
        )
