"""CICIDS2017 loader.

Reads the ``MachineLearningCSV`` bundle from the Canadian Institute for
Cybersecurity (8 day/attack CSVs, ~80 CICFlowMeter flow features). Cleaning:

* strip whitespace from column names (the CSVs ship with leading spaces);
* map the ``Label`` column to binary (BENIGN -> 0, any attack -> 1);
* drop leaky identifier columns (flow ID, IPs, ports, timestamp) so the model
  learns attack behaviour rather than host/service identity;
* replace +/-inf (e.g. ``Flow Bytes/s``) with NaN and drop affected rows;
* drop duplicate rows.

Reference: Sharafaldin, Lashkari & Ghorbani, ICISSP 2018.
Source: https://www.unb.ca/cic/datasets/ids-2017.html
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
    replace_inf,
    strip_column_names,
)

# Columns removed to avoid identity/leakage shortcuts. Destination Port is
# dropped by default because it effectively encodes the target service.
LEAKY_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Source Port",
    "Src Port",
    "Destination Port",
    "Dst Port",
    "Timestamp",
    "Fwd Header Length.1",  # duplicated column in the released CSVs
}

LABEL_CANDIDATES = ("label",)
BENIGN_TOKEN = "BENIGN"


@DATASETS.register("cicids2017")
class CICIDS2017Loader(RawLoader):
    name = "cicids2017"

    def __init__(self, data_dir: str = "data/raw/cicids2017", drop_destination_port: bool = True):
        super().__init__(data_dir)
        self.drop_destination_port = drop_destination_port

    def load_raw(self) -> RawData:
        files = self._find_files(["*.csv", "**/*.csv"])
        frames = [pd.read_csv(f, low_memory=False, skipinitialspace=True) for f in files]
        df = pd.concat(frames, ignore_index=True)
        df = strip_column_names(df)

        # Locate the label column (case-insensitive).
        label_col = next(
            (c for c in df.columns if c.lower() in LABEL_CANDIDATES), None
        )
        if label_col is None:
            raise ValueError(
                f"No label column found in CICIDS2017 files; columns: {list(df.columns)[:8]}..."
            )

        labels = df[label_col].astype("string").str.strip().str.upper()
        y = (labels != BENIGN_TOKEN).astype(np.int64).to_numpy()
        # Some CICIDS web-attack labels ship with a corrupted dash (the U+FFFD
        # replacement char is baked into the files); normalise it for readability.
        attack_names = sorted(
            n.replace("�", "-").replace("  ", " ").strip()
            for n in labels[labels != BENIGN_TOKEN].dropna().unique().tolist()
        )

        # Drop label + leaky identifier columns.
        drop = set(LEAKY_COLUMNS)
        if not self.drop_destination_port:
            drop -= {"Destination Port", "Dst Port"}
        features = drop_columns_ci(df.drop(columns=[label_col]), drop)

        # Numeric coercion, inf handling, then drop rows with any NaN.
        features = coerce_numeric(features)
        features = replace_inf(features)
        mask = ~features.isna().any(axis=1)
        features = features.loc[mask].reset_index(drop=True)
        y = y[mask.to_numpy()]

        # Drop duplicate rows (feature+label together) to avoid inflated counts.
        combined = features.copy()
        combined["__y__"] = y
        combined = combined.drop_duplicates().reset_index(drop=True)
        y = combined.pop("__y__").to_numpy()
        features = combined

        meta = {
            "source": "CICIDS2017 (CIC/UNB)",
            "n_files": len(files),
            "attack_classes": attack_names,
            "dropped_destination_port": self.drop_destination_port,
            "positive_label": "attack",
        }
        return RawData(
            X=features,
            y=y,
            feature_names=list(features.columns),
            label_name="label",
            meta=meta,
        )
