"""Base types and helpers for dataset loaders.

A *loader* handles the dataset-specific work: locating the raw files, reading
them, and cleaning them into a consistent form (binary labels, leaky identifier
columns removed, NaN/Inf handled, duplicates dropped). It returns a
:class:`RawData` bundle. Categorical encoding is intentionally deferred until
after the train/test split so its learned vocabulary cannot leak test data.

The shared, dataset-agnostic steps that come *after* this — train/test split,
scaler fitting, train-only balancing, freezing the test set, carving the attack
subset — are implemented separately (preprocessing pipeline) and consume a
``RawData``. Keeping the two apart means the genuinely per-dataset logic lives
here and the identical downstream pipeline is written once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RawData:
    """Cleaned features and binary labels before train-only transformations."""

    X: pd.DataFrame
    y: np.ndarray
    feature_names: list[str]
    label_name: str = "label"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y).ravel()
        if len(self.X) != len(self.y):
            raise ValueError("X and y have different numbers of rows.")
        if list(self.X.columns) != list(self.feature_names):
            raise ValueError("feature_names must match X columns exactly.")
        uniq = set(np.unique(self.y).tolist())
        if not uniq.issubset({0, 1}):
            raise ValueError(f"Labels must be binary 0/1; found {sorted(uniq)}.")

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


class RawLoader(ABC):
    """Abstract dataset-specific loader returning a :class:`RawData`."""

    name: str

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    @abstractmethod
    def load_raw(self) -> RawData:
        """Read and clean the raw files into a :class:`RawData` bundle."""
        raise NotImplementedError

    # -- helpers shared by concrete loaders ------------------------------- #

    def _find_files(self, patterns: list[str]) -> list[Path]:
        """Return raw files matching any glob pattern, or raise with guidance."""
        files: list[Path] = []
        for pat in patterns:
            files.extend(sorted(self.data_dir.glob(pat)))
        # de-duplicate while preserving order
        seen: set[Path] = set()
        unique = [f for f in files if not (f in seen or seen.add(f))]
        if not unique:
            raise FileNotFoundError(
                f"No files matching {patterns} under '{self.data_dir}'. "
                f"Download {self.name} manually into that folder (see README)."
            )
        return unique


# --------------------------------------------------------------------------- #
# Cleaning helpers (pure functions, shared across loaders)
# --------------------------------------------------------------------------- #


def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace from column names observed in CICIDS2017 dataset."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def drop_columns_ci(df: pd.DataFrame, names: set[str]) -> pd.DataFrame:
    """Drop columns matching ``names`` case-insensitively (if present)."""
    lower = {n.lower() for n in names}
    keep = [c for c in df.columns if c.lower() not in lower]
    return df[keep]


def replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN so a single dropna handles both."""
    return df.replace([np.inf, -np.inf], np.nan)


def coerce_numeric(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    """Coerce all columns except ``exclude`` to numeric (bad values -> NaN)."""
    exclude = exclude or set()
    df = df.copy()
    for c in df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def group_rare_categories(
    series: pd.Series, top_n: int | None = None, min_freq: float = 0.0,
    other_label: str = "other",
) -> pd.Series:
    """Collapse infrequent categories into a single ``other`` bucket.

    Keeps the ``top_n`` most frequent categories and/or those with relative
    frequency >= ``min_freq``; everything else becomes ``other``. Keeps
    high-cardinality fields (e.g. UNSW-NB15 ``proto``) from exploding one-hot.
    """
    s = series.astype("string").fillna(other_label)
    counts = s.value_counts()
    keep = set(counts.index)
    if top_n is not None:
        keep = set(counts.head(top_n).index)
    if min_freq > 0.0:
        frac = counts / counts.sum()
        keep &= set(frac[frac >= min_freq].index)
    return s.where(s.isin(keep), other=other_label)
