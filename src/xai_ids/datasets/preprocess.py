"""Shared preprocessing: turn a cleaned ``RawData`` into a frozen ``DataBundle``.

Dataset-agnostic steps, done once for every dataset:

1. stratified, seeded train/test split;
2. scaler fitted on **train only**, then applied to both (frozen/persisted);
3. **train-only** class balancing (test is never touched);
4. a frozen, seeded **attack subset** carved from the test set.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..config import ExperimentConfig
from ..interfaces.dataset import DataBundle
from ..registry import DATASETS
from .base import RawData


def _make_scaler(normalise: str):
    return {"standard": StandardScaler, "minmax": MinMaxScaler, "none": None}[normalise]


def _balance(X: np.ndarray, y: np.ndarray, how: str, seed: int):
    if how == "none":
        return X, y
    if how == "undersample_train_only":
        rng = np.random.default_rng(seed)
        n = np.unique(y, return_counts=True)[1].min()
        idx = np.concatenate([rng.choice(np.where(y == c)[0], n, replace=False)
                              for c in np.unique(y)])
        rng.shuffle(idx)
        return X[idx], y[idx]
    if how == "smote_train_only":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as e:
            raise ImportError(
                "balance='smote_train_only' needs imbalanced-learn (uv sync --extra ml)."
            ) from e
        return SMOTE(random_state=seed).fit_resample(X, y)
    raise ValueError(f"unknown balance strategy: {how}")


def preprocess(
    raw: RawData,
    *,
    seed: int = 42,
    test_size: float = 0.2,
    normalise: str = "standard",
    balance: str = "none",
    attack_subset_size: int = 500,
) -> DataBundle:
    X = raw.X.to_numpy(dtype=float)
    y = raw.y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler_cls = _make_scaler(normalise)
    scaler = None
    if scaler_cls is not None:
        scaler = scaler_cls().fit(X_train)  # fit on train ONLY
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    X_train, y_train = _balance(X_train, y_train, balance, seed)  # train only

    rng = np.random.default_rng(seed)
    n_sub = min(attack_subset_size, len(y_test))
    attack_indices = np.sort(rng.choice(len(y_test), n_sub, replace=False))

    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_attack=X_test[attack_indices],
        attack_indices=attack_indices,
        feature_names=raw.feature_names,
        scaler=scaler,
        meta=raw.meta,
    )


def build_dataset(config: ExperimentConfig) -> DataBundle:
    """Resolve the loader from config, read raw data, and preprocess it."""
    loader_cls = DATASETS.get(config.dataset.name)
    data_dir = config.dataset.data_dir
    loader = loader_cls() if data_dir is None else loader_cls(data_dir)
    config.dataset.data_dir = str(loader.data_dir)  # record the resolved path for provenance
    raw = loader.load_raw()
    d = config.dataset
    return preprocess(
        raw,
        seed=config.run.seed,
        test_size=d.test_size,
        normalise=d.normalise,
        balance=d.balance,
        attack_subset_size=d.attack_subset_size,
    )
