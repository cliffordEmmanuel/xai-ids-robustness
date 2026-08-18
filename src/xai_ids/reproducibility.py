"""Reproducibility utilities: seeding, run IDs, run directories, logging, provenance.

Every experiment should be re-runnable from its config plus the captured
provenance (seeds + pinned package versions). Timestamped run directories keep
outputs from colliding across runs and seeds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

_PROVENANCE_PACKAGES = (
    "numpy",
    "pyyaml",
    "pydantic",
    "scikit-learn",
    "xgboost",
    "torch",
    "shap",
    "lime",
    "pandas",
    "scipy",
    "imbalanced-learn",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch for deterministic runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional dependency (installed with the [ml] extra)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001 - absent or unusable torch must not break seeding
        pass


def generate_run_id(config: Any = None, *, seed: int | None = None) -> str:
    """Create a unique run ID: ``<UTC-timestamp>_<8-char config/seed hash>``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = ""
    if config is not None:
        # pydantic model -> stable JSON; fall back to repr for anything else.
        dump = getattr(config, "model_dump_json", None)
        payload = dump() if callable(dump) else repr(config)
    if seed is not None:
        payload += f"|seed={seed}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{stamp}_{digest}"


def make_run_dir(output_dir: str | Path, run_id: str) -> Path:
    """Create and return a timestamped run directory."""
    path = Path(output_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def capture_versions() -> dict[str, str]:
    """Return a dict of pinned versions for provenance (missing pkgs omitted)."""
    import platform
    import sys

    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in _PROVENANCE_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return versions


def capture_code_state() -> dict[str, str | int | bool | None]:
    """Record the checked-out revision and whether final runs used a dirty tree."""
    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "worktree_dirty": None, "changed_file_count": None}
    return {
        "git_revision": revision,
        "worktree_dirty": bool(status),
        "changed_file_count": len(status),
    }


def setup_logging(
    run_dir: str | Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure a logger that writes to stdout and (optionally) ``run.log``."""
    logger = logging.getLogger("xai_ids")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if run_dir is not None:
        file_handler = logging.FileHandler(Path(run_dir) / "run.log", mode="w")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def write_provenance(
    run_dir: str | Path,
    config: Any = None,
    seed: int | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write versions, config, and optional JSON-serialisable extra provenance."""
    run_dir = Path(run_dir)
    record: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "versions": capture_versions(),
        "code": capture_code_state(),
    }
    if config is not None:
        dump = getattr(config, "model_dump", None)
        record["config"] = dump(mode="json") if callable(dump) else repr(config)
    if extra:
        record["extra"] = extra

    out = run_dir / "provenance.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return out
