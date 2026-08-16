"""Save/load a preprocessed DataBundle so preprocessing runs once per config."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from ..interfaces.dataset import DataBundle


def save_bundle(bundle: DataBundle, path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "arrays.npz",
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        X_attack=bundle.X_attack,
        attack_indices=bundle.attack_indices,
    )
    if bundle.scaler is not None:
        joblib.dump(bundle.scaler, path / "scaler.joblib")
    (path / "meta.json").write_text(
        json.dumps(
            {
                "feature_names": bundle.feature_names,
                "meta": bundle.meta,
                "has_scaler": bundle.scaler is not None,
            },
            indent=2,
        )
    )
    return path


def load_bundle(path: str | Path) -> DataBundle:
    path = Path(path)
    a = np.load(path / "arrays.npz")
    info = json.loads((path / "meta.json").read_text())
    scaler = joblib.load(path / "scaler.joblib") if info["has_scaler"] else None
    return DataBundle(
        X_train=a["X_train"],
        y_train=a["y_train"],
        X_test=a["X_test"],
        y_test=a["y_test"],
        X_attack=a["X_attack"],
        attack_indices=a["attack_indices"],
        feature_names=info["feature_names"],
        scaler=scaler,
        meta=info["meta"],
    )
