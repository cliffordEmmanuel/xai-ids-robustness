"""Tests for the per-dataset loaders using small synthetic fixtures.

These mimic the real CICIDS2017 / UNSW-NB15 schemas (column names, dtypes,
NaN/Inf, duplicates, categoricals) so the loader logic is exercised without the
multi-GB real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import xai_ids.datasets  # noqa: F401 - triggers loader registration
from xai_ids.datasets.categorical import TrainOnlyCategoricalEncoder
from xai_ids.datasets.cicids2017 import CICIDS2017Loader
from xai_ids.datasets.unsw_nb15 import UNSWNB15Loader
from xai_ids.registry import DATASETS

# --------------------------------------------------------------------------- #
# CICIDS2017
# --------------------------------------------------------------------------- #


def _write_cicids_fixture(dir_) -> None:
    # Column names carry leading spaces (as in the released CSVs).
    rows = [
        # Flow ID, Destination Port, Flow Duration, Total Fwd Packets, Flow Bytes/s, Label
        ["id1", 80, 100, 10, 1000.0, "BENIGN"],
        ["id2", 443, 200, 20, 2000.0, "DoS Hulk"],
        ["id3", 22, 300, 30, np.inf, "PortScan"],       # inf -> dropped
        ["id4", 80, 100, 10, 1000.0, "BENIGN"],          # dup of row 1 after id/port drop
        ["id5", 21, 400, 40, 4000.0, "BENIGN"],
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "Flow ID",
            " Destination Port",
            " Flow Duration",
            " Total Fwd Packets",
            " Flow Bytes/s",
            " Label",
        ],
    )
    df.to_csv(dir_ / "day1.csv", index=False)


def test_cicids2017_loader(tmp_path):
    _write_cicids_fixture(tmp_path)
    raw = CICIDS2017Loader(data_dir=tmp_path).load_raw()

    # Leaky identifier columns removed.
    assert "Flow ID" not in raw.feature_names
    assert "Destination Port" not in raw.feature_names
    assert raw.feature_names == ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s"]

    # inf row dropped, duplicate dropped -> 3 rows (2 benign, 1 attack).
    assert raw.n_samples == 3
    assert set(np.unique(raw.y).tolist()) == {0, 1}
    assert raw.y.sum() == 1

    # No NaN / Inf survive.
    assert not raw.X.isna().any().any()
    assert np.isfinite(raw.X.to_numpy()).all()
    assert "DOS HULK" in raw.meta["attack_classes"]


def test_cicids2017_keep_destination_port(tmp_path):
    _write_cicids_fixture(tmp_path)
    raw = CICIDS2017Loader(data_dir=tmp_path, drop_destination_port=False).load_raw()
    assert "Destination Port" in raw.feature_names


# --------------------------------------------------------------------------- #
# UNSW-NB15
# --------------------------------------------------------------------------- #


def _write_unsw_fixture(dir_) -> None:
    cols = ["id", "dur", "proto", "service", "state", "spkts", "attack_cat", "label"]
    train = pd.DataFrame(
        [
            [1, 0.1, "tcp", "http", "FIN", 10, "Normal", 0],
            [2, 0.2, "udp", "dns", "CON", 5, "Exploits", 1],
        ],
        columns=cols,
    )
    test = pd.DataFrame(
        [
            [3, 0.3, "tcp", "-", "INT", 8, "Normal", 0],
            [4, 0.4, "arp", "http", "FIN", 2, "Fuzzers", 1],
        ],
        columns=cols,
    )
    train.to_csv(dir_ / "UNSW_NB15_training-set.csv", index=False)
    test.to_csv(dir_ / "UNSW_NB15_testing-set.csv", index=False)


def test_unsw_nb15_loader(tmp_path):
    _write_unsw_fixture(tmp_path)
    raw = UNSWNB15Loader(data_dir=tmp_path).load_raw()

    # id / attack_cat / label removed from features.
    for dropped in ("id", "attack_cat", "label"):
        assert dropped not in raw.feature_names

    # Categoricals remain raw until the train-only preprocessor fits an encoder.
    for categorical in ("proto", "service", "state"):
        assert categorical in raw.feature_names
    assert raw.meta["categorical_encoding"]["columns"] == ["proto", "service", "state"]

    # Numeric passthrough kept.
    assert "dur" in raw.feature_names and "spkts" in raw.feature_names

    # Labels binary and correct.
    assert raw.n_samples == 4
    assert raw.y.tolist() == [0, 1, 0, 1]
    assert not raw.X.isna().any().any()
    assert "Exploits" in raw.meta["attack_categories"]


def test_train_only_categorical_encoder_maps_test_only_category_to_other():
    train = pd.DataFrame(
        {
            "dur": [0.1, 0.2, 0.3, 0.4],
            "proto": ["tcp", "tcp", "udp", "udp"],
        }
    )
    test = pd.DataFrame({"dur": [0.5], "proto": ["test_only"]})
    encoder = TrainOnlyCategoricalEncoder(columns=("proto",), top_n={"proto": 2})

    encoded_train = encoder.fit_transform(train)
    encoded_test = encoder.transform(test)

    assert "proto_test_only" not in encoder.output_columns_
    assert "proto_other" in encoder.output_columns_
    assert encoded_train.columns.tolist() == encoded_test.columns.tolist()
    assert encoded_test.loc[0, "proto_other"] == 1


# --------------------------------------------------------------------------- #
# Registry + error handling
# --------------------------------------------------------------------------- #


def test_loaders_registered():
    assert "cicids2017" in DATASETS
    assert "unsw_nb15" in DATASETS
    assert DATASETS.get("cicids2017") is CICIDS2017Loader
    assert DATASETS.get("unsw_nb15") is UNSWNB15Loader


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="manually"):
        CICIDS2017Loader(data_dir=tmp_path / "empty").load_raw()
