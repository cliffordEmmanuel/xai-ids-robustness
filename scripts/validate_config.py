#!/usr/bin/env python3
"""Validate one or more experiment config files against the schema.

Usage:
    python scripts/validate_config.py configs/fragility_example.yaml [more.yaml ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_ids.config import load_config  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    rc = 0
    for path in argv:
        try:
            cfg = load_config(path)
        except Exception as exc:  # noqa: BLE001 - surface any validation error
            print(f"[INVALID] {path}\n    {exc}\n")
            rc = 1
        else:
            print(
                f"[OK]      {path}  "
                f"(strand={cfg.attack.strand}, model={cfg.model.name}, "
                f"explainer={cfg.explainer.name}, dataset={cfg.dataset.name})"
            )
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
