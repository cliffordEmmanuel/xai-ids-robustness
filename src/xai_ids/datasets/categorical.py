"""Train-only categorical encoding for tabular IDS datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TrainOnlyCategoricalEncoder:
    """Rare-category grouper and one-hot encoder fitted on training rows only."""

    columns: tuple[str, ...]
    top_n: dict[str, int]
    other_label: str = "other"
    categories_: dict[str, list[str]] = field(default_factory=dict)
    output_columns_: list[str] = field(default_factory=list)

    def _normalise(self, series: pd.Series) -> pd.Series:
        return series.astype("string").fillna(self.other_label)

    def fit(self, frame: pd.DataFrame) -> TrainOnlyCategoricalEncoder:
        """Learn category retention and output columns from training rows only."""
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"categorical columns missing during fit: {sorted(missing)}")

        self.categories_ = {}
        for column in self.columns:
            counts = self._normalise(frame[column]).value_counts()
            ordered = sorted(counts.index.astype(str), key=lambda value: (-counts[value], value))
            retained = ordered[: self.top_n[column]]
            if self.other_label not in retained:
                retained.append(self.other_label)
            self.categories_[column] = retained

        self.output_columns_ = self._encode(frame).columns.astype(str).tolist()
        return self

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode a frame using fitted categories without learning from that frame."""
        if not self.categories_:
            raise RuntimeError("TrainOnlyCategoricalEncoder must be fitted before transform.")

        encoded = frame.drop(columns=list(self.columns)).copy()
        for column in self.columns:
            allowed = set(self.categories_[column])
            values = self._normalise(frame[column])
            values = values.where(values.isin(allowed), other=self.other_label)
            category = pd.Categorical(values, categories=self.categories_[column])
            dummies = pd.get_dummies(category, prefix=column, dtype=np.int64)
            dummies.index = frame.index
            encoded = pd.concat([encoded, dummies], axis=1)
        return encoded

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the training-derived representation to new rows."""
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"categorical columns missing during transform: {sorted(missing)}")
        encoded = self._encode(frame)
        return encoded.reindex(columns=self.output_columns_, fill_value=0)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def state(self) -> dict:
        """JSON-serialisable fitted state for provenance and cache fingerprints."""
        return {
            "columns": list(self.columns),
            "top_n": self.top_n,
            "other_label": self.other_label,
            "categories": self.categories_,
            "output_columns": self.output_columns_,
        }


def categorical_encoder_from_meta(meta: dict) -> TrainOnlyCategoricalEncoder | None:
    """Build an unfitted transformer from optional dataset metadata."""
    encoding = meta.get("categorical_encoding")
    if not encoding:
        return None
    return TrainOnlyCategoricalEncoder(
        columns=tuple(encoding["columns"]),
        top_n={name: int(limit) for name, limit in encoding["top_n"].items()},
        other_label=encoding.get("other_label", "other"),
    )
