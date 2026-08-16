"""LIME explainer (model-agnostic; identical for every architecture).

Fits a local linear surrogate per instance from ``predict_proba`` queries around
each point. Needs a background sample for feature statistics, is stochastic
(seeded), and returns per-raw-feature weights for the positive (attack) class as
``(n, d)``. ``lime`` is imported lazily so the core CLI works without the ml extra.
"""

from __future__ import annotations

import numpy as np

from ..interfaces.explainer import Explainer
from ..registry import EXPLAINERS


@EXPLAINERS.register("lime")
class LIMEExplainer(Explainer):
    name = "lime"

    def __init__(
        self,
        num_samples: int = 1500,
        background_size: int = 1000,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.background_size = background_size
        self.seed = seed
        self._explainer = None  # cached across calls

    def _sample(self, bg: np.ndarray) -> np.ndarray:
        bg = np.asarray(bg, dtype=float)
        if len(bg) > self.background_size:
            rng = np.random.default_rng(self.seed)
            bg = bg[rng.choice(len(bg), self.background_size, replace=False)]
        return bg

    def explain(self, model, X, background=None) -> np.ndarray:
        from lime.lime_tabular import LimeTabularExplainer

        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if self._explainer is None:
            bg = self._sample(background if background is not None else X)
            self._explainer = LimeTabularExplainer(
                bg,
                mode="classification",
                feature_names=[str(i) for i in range(d)],
                discretize_continuous=False,  # weights in raw feature space (like SHAP)
                random_state=self.seed,
            )

        out = np.zeros((n, d), dtype=float)
        for i, x in enumerate(X):
            exp = self._explainer.explain_instance(
                x,
                model.predict_proba,
                num_features=d,
                num_samples=self.num_samples,
                labels=(1,),
            )
            for fidx, weight in exp.local_exp[1]:  # weights for the attack class
                out[i, fidx] = weight
        return out
