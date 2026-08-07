"""Metric interface.

Robustness metrics (Phase 6) compare baseline vs adversarial attributions.
Fidelity metrics (deletion/insertion, local fidelity) query the model, so
``compute`` accepts an optional ``model`` — this is a deliberate Path B change
from the original ``compute(baseline, adversarial)`` signature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .model import Model


class Metric(ABC):
    """Abstract robustness metric."""

    name: str
    #: whether ``compute`` requires the ``model`` argument (e.g. fidelity metrics)
    requires_model: bool = False

    @abstractmethod
    def compute(
        self,
        baseline: np.ndarray,
        adversarial: np.ndarray,
        *,
        model: Model | None = None,
    ) -> float | dict[str, float]:
        """Return a scalar (or named scalars) comparing the two attribution sets."""
        raise NotImplementedError
