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

    feature_names: list[str]
    scaler: Any = None  # fitted on train only, frozen, persisted
    categorical_transformer: Any = None  # fitted on train only, frozen, persisted
    fidelity_baseline: np.ndarray | None = None  # derived from train only, model space
    feature_groups: list[list[int]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("Train/test feature dimensionality mismatch.")
        if len(self.feature_names) != self.X_train.shape[1]:
            raise ValueError(
                "feature_names length does not match the number of features."
            )
        if self.fidelity_baseline is not None and self.fidelity_baseline.shape != (
            self.X_train.shape[1],
        ):
            raise ValueError("fidelity_baseline must have one value per feature.")
        if self.feature_groups:
            flattened = [index for group in self.feature_groups for index in group]
            expected = list(range(self.X_train.shape[1]))
            if sorted(flattened) != expected:
                raise ValueError("feature_groups must partition every feature exactly once.")

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def feature_index(self) -> dict[str, int]:
        """Map feature name -> column index (for the tamper target rule)."""
        return {name: i for i, name in enumerate(self.feature_names)}


class Dataset(ABC):
    """Abstract dataset loader."""

    name: str

    @abstractmethod
    def load(self) -> DataBundle:
        """Load, clean, normalise, split, and freeze the dataset."""
        raise NotImplementedError
