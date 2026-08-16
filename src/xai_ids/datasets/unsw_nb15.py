"""UNSW-NB15 loader.

Reads the pre-partitioned ``UNSW_NB15_training-set.csv`` / ``testing-set.csv``
(~45 columns, binary ``label`` plus multiclass ``attack_cat``). Cleaning:

* drop the ``id`` row index and the ``attack_cat`` column (kept only in meta for
  later stratification);
* one-hot encode the categorical fields (``proto``, ``service``, ``state``),
  collapsing rare/high-cardinality values (notably ``proto``) into ``other`` so
  the encoding does not explode into hundreds of columns;
* coerce remaining columns to numeric, handle NaN/Inf, drop duplicates.

Reference: Moustafa & Slay, MilCIS 2015 (and companion papers).
Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import DATASETS
from .base import (
    RawData,
    RawLoader,
    coerce_numeric,
    drop_columns_ci,
    group_rare_categories,
    replace_inf,
    strip_column_names,
)

CATEGORICAL_COLUMNS = ("proto", "service", "state")
DROP_COLUMNS = {"id"}
LABEL_COLUMN = "label"
ATTACK_CAT_COLUMN = "attack_cat"

# High-cardinality caps for one-hot encoding.
PROTO_TOP_N = 15
SERVICE_TOP_N = 15
STATE_TOP_N = 12


@DATASETS.register("unsw_nb15")
class UNSWNB15Loader(RawLoader):
    name = "unsw_nb15"

    def __init__(self, data_dir: str = "data/raw/unsw_nb15"):
        super().__init__(data_dir)

    def load_raw(self) -> RawData:
        files = self._find_files(
            ["*training-set.csv", "*testing-set.csv", "UNSW_NB15_*.csv", "*.csv"]
        )
        frames = [pd.read_csv(f, low_memory=False) for f in files]
        df = pd.concat(frames, ignore_index=True)
        df = strip_column_names(df)
        df.columns = [c.lower() for c in df.columns]  # normalise to lowercase

        if LABEL_COLUMN not in df.columns:
            raise ValueError(
                f"No '{LABEL_COLUMN}' column in UNSW-NB15 files; columns: {list(df.columns)[:8]}..."
            )

        y = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
        attack_cats = (
            sorted(df[ATTACK_CAT_COLUMN].dropna().astype(str).unique().tolist())
            if ATTACK_CAT_COLUMN in df.columns
            else []
        )

        # Remove label, attack_cat, and id from the feature frame.
        df = drop_columns_ci(df, DROP_COLUMNS | {LABEL_COLUMN, ATTACK_CAT_COLUMN})

        # One-hot encode categoricals with rare-category grouping.
        caps = {"proto": PROTO_TOP_N, "service": SERVICE_TOP_N, "state": STATE_TOP_N}
        present_cats = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        for c in present_cats:
            df[c] = group_rare_categories(df[c], top_n=caps[c])
        if present_cats:
            df = pd.get_dummies(df, columns=present_cats, prefix=present_cats, dtype=np.int64)

        # Numeric coercion + inf/NaN handling on the remaining columns.
        df = coerce_numeric(df)
        df = replace_inf(df)

        combined = df.copy()
        combined["__y__"] = y.to_numpy()
        mask = ~combined.isna().any(axis=1)
        combined = combined.loc[mask].reset_index(drop=True)
        combined = combined.drop_duplicates().reset_index(drop=True)
        y_final = combined.pop("__y__").astype(np.int64).to_numpy()
        features = combined

        meta = {
            "source": "UNSW-NB15 (UNSW Canberra)",
            "n_files": len(files),
            "attack_categories": attack_cats,
            "encoded_categoricals": present_cats,
            "positive_label": "attack",
        }
        return RawData(
            X=features,
            y=y_final,
            feature_names=list(features.columns),
            label_name="label",
            meta=meta,
        )
