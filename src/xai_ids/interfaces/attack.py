"""Attack interface.

The single attack framework (Phase 5) covers both strands via a pluggable
objective and gradient source:

* ``objective='loss'``        -> fragility strand (standard FGSM/PGD).
* ``objective='attribution'`` -> tamper strand (targeted; requires an explanation oracle).
* ``gradient_source='spsa'``  -> zeroth-order estimation, both models (default).
* ``gradient_source='autograd'`` -> MLP-only optional sanity check.

For the tamper strand the attack calls an explanation service inside its
optimisation loop. ``generate`` accepts restricted oracle interfaces rather than
the raw model, explainer, or training background.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..oracles import ExplanationOracle, ProbabilityOracle

Objective = Literal["loss", "attribution"]
GradientSource = Literal["spsa", "autograd"]


@dataclass
class AttackResult:
    """Output of an attack over a batch of inputs."""

    X_adv: np.ndarray
    prediction_preserved: np.ndarray  # bool mask, shape (n_samples,)
    n_queries: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prediction_preserved.shape[0] != self.X_adv.shape[0]:
            raise ValueError("prediction_preserved mask must align with X_adv rows.")

    @property
    def retention_rate(self) -> float:
        """Fraction of samples whose prediction was preserved."""
        if self.prediction_preserved.size == 0:
            return 0.0
        return float(self.prediction_preserved.mean())


class Attack(ABC):
    """Abstract adversarial attack."""

    name: str

    @abstractmethod
    def generate(
        self,
        model_oracle: ProbabilityOracle,
        X: np.ndarray,
        *,
        objective: Objective = "loss",
        gradient_source: GradientSource = "spsa",
        explanation_oracle: ExplanationOracle | None = None,
    ) -> AttackResult:
        """Generate adversarial inputs for ``X``.

        Implementations must raise if ``objective='attribution'`` is requested
        without an ``explanation_oracle``. Implementations must interact only
        with these restricted oracle interfaces, never with a raw model or
        explainer object.
        """
        raise NotImplementedError
