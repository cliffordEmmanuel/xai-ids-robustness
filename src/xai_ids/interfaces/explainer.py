"""Explainer interface.

SHAP and LIME implement this in Phase 4. Because the tamper strand calls the
explainer *inside* the attack loop, implementations must be fast (TreeSHAP for
XGBoost; capped-sample KernelSHAP/LIME) and expose helpers for the target rule
and the LIME noise floor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .model import Model


class Explainer(ABC):
    """Abstract post-hoc feature-attribution explainer."""

    name: str

    @abstractmethod
    def explain(self, model: Model, X: np.ndarray) -> np.ndarray:
        """Return an attribution array of shape ``(n_samples, n_features)``."""
        raise NotImplementedError

    def target_features(self, attributions: np.ndarray, k: int = 1) -> np.ndarray:
        """Indices of the top-``k`` attributed features per sample.

        Used by the tamper strand's ``top1_suppress`` / ``topk_suppress`` rules.
        Default is a generic, model-agnostic implementation over ``|attribution|``.
        """
        order = np.argsort(-np.abs(attributions), axis=1)
        return order[:, :k]

    def noise_floor(
        self, model: Model, X: np.ndarray, repeats: int = 10
    ) -> np.ndarray:
        """Estimate intrinsic attribution variance on unperturbed inputs.

        Stochastic explainers (LIME) override or rely on this to separate
        intrinsic instability (Rashomon effect) from attack-induced drift.
        Deterministic explainers may return zeros. Default: repeated calls.
        """
        runs = np.stack([self.explain(model, X) for _ in range(repeats)], axis=0)
        return runs.std(axis=0)
