"""xai-ids: benchmarking the tamper resistance of XAI methods in ML-based IDS."""

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
