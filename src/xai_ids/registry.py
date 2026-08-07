"""A tiny name -> component registry.

Lets configs refer to components by string name (e.g. ``model: xgboost``) and the
runner resolve them without hard-coded imports. Concrete components register
themselves with the appropriate decorator as they are implemented in later phases.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class Registry:
    """A simple string-keyed registry for one component kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered.")
            self._items[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        if name not in self._items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Registered: {sorted(self._items)}"
            )
        return self._items[name]

    def available(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


# One registry per component kind.
DATASETS = Registry("dataset")
MODELS = Registry("model")
EXPLAINERS = Registry("explainer")
ATTACKS = Registry("attack")
METRICS = Registry("metric")
