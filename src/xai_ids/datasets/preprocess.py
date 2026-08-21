"""Shared preprocessing: turn a cleaned ``RawData`` into a frozen ``DataBundle``.

Dataset-agnostic steps, done once for every dataset:

1. stratified, seeded train/test split;
2. categorical encoding fitted on **train only**, then applied to both;
3. scaler fitted on **train only**, then applied to both (frozen/persisted);
4. **train-only** class balancing (test is never touched).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..config import ExperimentConfig
from ..interfaces.dataset import DataBundle
from ..registry import DATASETS
from .base import RawData
from .categorical import categorical_encoder_from_meta

PREPROCESSING_SCHEMA_VERSION = "2"


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


def _feature_groups(feature_names: list[str], categorical_transformer) -> list[list[int]]:
    """Partition feature indices, preserving one-hot categorical groups."""
    grouped: list[list[int]] = []
    assigned: set[int] = set()
    index = {name: i for i, name in enumerate(feature_names)}
    if categorical_transformer is not None:
        for column in categorical_transformer.columns:
            categories = categorical_transformer.categories_[column]
            names = [f"{column}_{category}" for category in categories]
            group = [index[name] for name in names if name in index]
            if group:
                grouped.append(group)
                assigned.update(group)
    grouped.extend([[i] for i in range(len(feature_names)) if i not in assigned])
    return grouped


def _training_reference(X_train: np.ndarray, feature_groups: list[list[int]]) -> np.ndarray:
    """Create a valid training-derived reference, including one-hot group modes."""
    reference = np.mean(X_train, axis=0)
    for group in feature_groups:
        if len(group) > 1:
            values = X_train[:, group].mean(axis=0)
            reference[group] = 0.0
            reference[group[int(np.argmax(values))]] = 1.0
    return reference


def preprocess(
    raw: RawData,
    *,
    seed: int = 42,
    test_size: float = 0.2,
    normalise: str = "standard",
    balance: str = "none",
) -> DataBundle:
    X_train_frame, X_test_frame, y_train, y_test = train_test_split(
        raw.X, raw.y, test_size=test_size, random_state=seed, stratify=raw.y
    )

    categorical_transformer = categorical_encoder_from_meta(raw.meta)
    if categorical_transformer is not None:
        X_train_frame = categorical_transformer.fit_transform(X_train_frame)
        X_test_frame = categorical_transformer.transform(X_test_frame)
        feature_names = categorical_transformer.output_columns_
    else:
        feature_names = raw.feature_names

    X_train = X_train_frame.to_numpy(dtype=float)
    X_test = X_test_frame.to_numpy(dtype=float)
    feature_groups = _feature_groups(feature_names, categorical_transformer)
    fidelity_baseline = _training_reference(X_train, feature_groups)

    scaler_cls = _make_scaler(normalise)
    scaler = None
    if scaler_cls is not None:
        scaler = scaler_cls().fit(X_train)  # fit on train ONLY
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        fidelity_baseline = scaler.transform(fidelity_baseline.reshape(1, -1))[0]

    X_train, y_train = _balance(X_train, y_train, balance, seed)  # train only

    # The attack subset is selected at attack time as true positives (needs the
    # trained model), so it is not carved here.
    meta = dict(raw.meta)
    meta["preprocessing_schema_version"] = PREPROCESSING_SCHEMA_VERSION
    meta["fidelity_reference"] = {
        "source": "training_mean_with_categorical_group_modes",
        "feature_groups": feature_groups,
    }
    if categorical_transformer is not None:
        meta["categorical_transformer"] = categorical_transformer.state()

    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
        categorical_transformer=categorical_transformer,
        fidelity_baseline=fidelity_baseline,
        feature_groups=feature_groups,
        meta=meta,
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
    )
