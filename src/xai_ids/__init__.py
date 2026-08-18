"""xai-ids: benchmarking the tamper resistance of XAI methods in ML-based IDS."""

# On macOS, XGBoost's bundled OpenMP runtime can clash with a second copy loaded
# via other wheels (numpy/scikit-learn) or Homebrew's libomp, which segfaults
# during training. Forcing a single OpenMP thread avoids the clash. This must run
# before numpy/xgboost are imported, so it lives at the top of the package. It is
# a default only: set OMP_NUM_THREADS yourself (e.g. export OMP_NUM_THREADS=8) to
# opt back into multithreading once your environment is known-good.
import os as _os

_os.environ.setdefault("OMP_NUM_THREADS", "1")

from .config import ExperimentConfig, load_config, load_config_from_dict
from .reproducibility import (
    capture_versions,
    generate_run_id,
    make_run_dir,
    set_seed,
    setup_logging,
    write_provenance,
)

__version__ = "0.1.0"

__all__ = [
    "ExperimentConfig",
    "load_config",
    "load_config_from_dict",
    "set_seed",
    "generate_run_id",
    "make_run_dir",
    "capture_versions",
    "setup_logging",
    "write_provenance",
    "__version__",
]
