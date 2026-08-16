"""Model interface.

Both IDS classifiers (XGBoost, MLP) implement this interface in Phase 3. The key
Path B constraint: ``predict_proba`` is the black-box attack surface, so it must
be available on every model regardless of whether the model exposes gradients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Model(ABC):
    """Abstract IDS classifier."""

    name: str
    ext: str = ".bin"  # on-disk model file extension (overridden per model)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard-label predictions of shape ``(n_samples,)``."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape ``(n_samples, n_classes)``.

        This is the query interface used by the (black-box, SPSA-based) attacks.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Model:
        raise NotImplementedError
