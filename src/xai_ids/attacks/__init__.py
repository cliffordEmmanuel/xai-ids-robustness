"""Attacks. Importing this package registers them in ``ATTACKS``."""

from ..registry import ATTACKS
from .pgd import PGDAttack

__all__ = ["ATTACKS", "PGDAttack"]
