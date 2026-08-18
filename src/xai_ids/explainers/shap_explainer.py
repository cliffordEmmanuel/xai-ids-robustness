"""SHAP explainer with per-architecture backend dispatch.

* ``backend='tree'`` -> TreeSHAP: exact, fast, for XGBoost; needs ``model.native``.
* ``backend='kernel'`` -> KernelSHAP: model-agnostic approximation for the MLP;
  uses ``predict_proba`` + a background sample.

Attributions are returned for the positive (attack) class as ``(n, d)``.
``shap`` is imported lazily so the core CLI works without the ml extra.
"""

from __future__ import annotations

from contextlib import contextmanager

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
        kernel_l1_reg: float | str = 0.0001,
        seed: int = 42,
        kernel_random_seed: int | None = None,
    ):
        self.backend = backend
        self.background_size = background_size
        self.kernel_nsamples = kernel_nsamples
        self.kernel_l1_reg = kernel_l1_reg
        self.seed = seed
        self.kernel_random_seed = seed if kernel_random_seed is None else kernel_random_seed
        self._explainer = None  # cached across calls (reused in the attack loop)
        self._kernel_explain_calls = 0
        self._diagnostic_calls: list[dict[str, float | int | bool]] = []

    @staticmethod
    @contextmanager
    def _temporary_numpy_seed(seed: int):
        """Run KernelSHAP's NumPy sampling reproducibly without leaking global state."""
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)

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

    @staticmethod
    def _positive_expected_value(value) -> float:
        values = np.asarray(value, dtype=float).reshape(-1)
        if not len(values):
            raise ValueError("KernelSHAP did not expose an expected value")
        return float(values[1] if len(values) > 1 else values[0])

    def _record_kernel_diagnostics(self, model, X: np.ndarray, attrs: np.ndarray) -> None:
        """Track finite values and local-accuracy residuals for every KernelSHAP call."""
        finite = bool(np.isfinite(attrs).all())
        if finite:
            expected = self._positive_expected_value(self._explainer.expected_value)
            reconstructed = expected + attrs.sum(axis=1)
            probabilities = np.asarray(model.predict_proba(X), dtype=float)[:, 1]
            residuals = np.abs(reconstructed - probabilities)
            max_error = float(np.max(residuals)) if len(residuals) else 0.0
            mean_error = float(np.mean(residuals)) if len(residuals) else 0.0
        else:
            max_error = float("inf")
            mean_error = float("inf")
        self._diagnostic_calls.append(
            {
                "n_rows": int(len(X)),
                "attributions_finite": finite,
                "max_additivity_error": max_error,
                "mean_additivity_error": mean_error,
            }
        )

    def numerical_diagnostics(self) -> dict[str, float | int | bool]:
        """Return cumulative evaluator-side diagnostics for this explainer instance."""
        if self.backend != "kernel":
            return {
                "calls": 0,
                "attributions_finite": True,
                "max_additivity_error": 0.0,
                "mean_additivity_error": 0.0,
            }
        if not self._diagnostic_calls:
            return {
                "calls": 0,
                "attributions_finite": True,
                "max_additivity_error": 0.0,
                "mean_additivity_error": 0.0,
            }
        return {
            "calls": len(self._diagnostic_calls),
            "attributions_finite": all(
                bool(record["attributions_finite"]) for record in self._diagnostic_calls
            ),
            "max_additivity_error": float(
                max(float(record["max_additivity_error"]) for record in self._diagnostic_calls)
            ),
            "mean_additivity_error": float(
                np.mean(
                    [float(record["mean_additivity_error"]) for record in self._diagnostic_calls]
                )
            ),
        }

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
            call_seed = self.kernel_random_seed + self._kernel_explain_calls
            self._kernel_explain_calls += 1
            with self._temporary_numpy_seed(call_seed):
                vals = self._explainer.shap_values(
                    X,
                    nsamples=self.kernel_nsamples,
                    l1_reg=self.kernel_l1_reg,
                    silent=True,
                )
        attrs = self._positive_2d(vals)
        if self.backend == "kernel":
            self._record_kernel_diagnostics(model, X, attrs)
        return attrs
