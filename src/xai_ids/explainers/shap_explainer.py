"""SHAP explainer with per-architecture backend dispatch.

* ``backend='tree'`` -> TreeSHAP: exact, fast, for XGBoost; needs ``model.native``.
* ``backend='kernel'`` -> KernelSHAP: model-agnostic approximation for the MLP;
  uses ``predict_proba`` + a background sample.

Attributions are returned for the positive (attack) class as ``(n, d)``.
``shap`` is imported lazily so the core CLI works without the ml extra.
"""

from __future__ import annotations

import numpy as np

from ..interfaces.explainer import Explainer
from ..registry import EXPLAINERS


@EXPLAINERS.register("shap")
class SHAPExplainer(Explainer):
    name = "shap"

    def __init__(
        self,
        backend: str = "tree",
        background_size: int = 100,
        kernel_nsamples: int = 100,
        seed: int = 42,
    ):
        self.backend = backend
        self.background_size = background_size
        self.kernel_nsamples = kernel_nsamples
        self.seed = seed
        self._explainer = None  # cached across calls (reused in the attack loop)

    def _sample(self, bg: np.ndarray) -> np.ndarray:
        bg = np.asarray(bg, dtype=float)
        if len(bg) > self.background_size:
            rng = np.random.default_rng(self.seed)
            bg = bg[rng.choice(len(bg), self.background_size, replace=False)]
        return bg

    @staticmethod
    def _positive_2d(vals) -> np.ndarray:
        """Normalise SHAP output (version-dependent) to (n, d) for the attack class."""
        if isinstance(vals, list):  # [class0, class1]
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, d, n_classes)
            vals = vals[..., 1]
        return vals

    def explain(self, model, X, background=None) -> np.ndarray:
        import shap

        X = np.asarray(X, dtype=float)
        if self.backend == "tree":
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(model.native)
            vals = self._explainer.shap_values(X)
        else:  # kernel
            if self._explainer is None:
                bg = self._sample(background if background is not None else X)
                self._explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:, 1], bg
                )
            vals = self._explainer.shap_values(
                X, nsamples=self.kernel_nsamples, silent=True
            )
        return self._positive_2d(vals)
