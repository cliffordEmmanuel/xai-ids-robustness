"""XGBoost IDS classifier.

Class imbalance is handled with ``scale_pos_weight`` (neg/pos ratio) rather than
resampling. ``xgboost`` is imported lazily so the core CLI works without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..interfaces.model import Model
from ..registry import MODELS

_DEFAULTS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "tree_method": "hist",
    "n_jobs": -1,
    "eval_metric": "logloss",
}


@MODELS.register("xgboost")
class XGBoostModel(Model):
    name = "xgboost"
    ext = ".ubj"  # native XGBoost UBJSON (portable, no pickle)

    def __init__(self, class_weight: bool = True, params: dict | None = None):
        self.class_weight = class_weight
        self.params = {**_DEFAULTS, **(params or {})}
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostModel:
        import xgboost as xgb

        spw = 1.0
        if self.class_weight:
            pos, neg = float((y == 1).sum()), float((y == 0).sum())
            spw = neg / pos if pos > 0 else 1.0
        self._model = xgb.XGBClassifier(scale_pos_weight=spw, **self.params)
        self._model.fit(X, y)
        return self

    @property
    def native(self):
        """Underlying XGBClassifier (needed by TreeSHAP)."""
        return self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))  # native format inferred from extension

    @classmethod
    def load(cls, path: str | Path) -> XGBoostModel:
        import xgboost as xgb

        obj = cls()
        obj._model = xgb.XGBClassifier()
        obj._model.load_model(str(path))
        return obj
