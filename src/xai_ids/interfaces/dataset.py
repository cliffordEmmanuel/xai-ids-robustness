"""Dataset interface.

Concrete datasets (CICIDS2017, UNSW-NB15) are implemented in Phase 2. A loader
returns a fully prepared :class:`DataBundle`: train/test arrays, the frozen
attack subset used by the (expensive) tamper strand, the fitted scaler, and the
feature-name list needed by the dataset-agnostic target rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DataBundle:
    """Prepared, frozen dataset ready for modelling and attacks."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    # Frozen slice of the test set used by the tamper strand.
    X_attack: np.ndarray
    attack_indices: np.ndarray

    feature_names: list[str]
    scaler: Any = None  # fitted on train only, frozen, persisted
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("Train/test feature dimensionality mismatch.")
        if len(self.feature_names) != self.X_train.shape[1]:
            raise ValueError(
                "feature_names length does not match the number of features."
            )

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


class Dataset(ABC):
    """Abstract dataset loader."""

    name: str

    @abstractmethod
    def load(self) -> DataBundle:
        """Load, clean, normalise, split, and freeze the dataset."""
        raise NotImplementedError
