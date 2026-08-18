"""IDS models. Importing this package registers the models in ``MODELS``."""

from ..registry import MODELS
from .evaluate import assess_model_quality, evaluate_classification
from .mlp_model import MLPModel
from .xgboost_model import XGBoostModel

__all__ = [
    "MODELS",
    "assess_model_quality",
    "evaluate_classification",
    "MLPModel",
    "XGBoostModel",
]
