"""Dataset loaders.

Importing this package registers the concrete loaders in the ``DATASETS``
registry, so ``DATASETS.get("cicids2017")`` / ``DATASETS.get("unsw_nb15")``
resolve after import.
"""

from ..registry import DATASETS
from .base import RawData, RawLoader
from .cicids2017 import CICIDS2017Loader
from .preprocess import build_dataset, preprocess
from .store import load_bundle, save_bundle
from .unsw_nb15 import UNSWNB15Loader

__all__ = [
    "DATASETS",
    "RawData",
    "RawLoader",
    "CICIDS2017Loader",
    "UNSWNB15Loader",
    "preprocess",
    "build_dataset",
    "save_bundle",
    "load_bundle",
]
