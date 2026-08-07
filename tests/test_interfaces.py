"""Tests for the abstract interfaces, supporting dataclasses, and registry."""

from __future__ import annotations

import numpy as np
import pytest

from xai_ids.interfaces import (
    Attack,
    AttackResult,
    DataBundle,
    Dataset,
    Explainer,
    Metric,
    Model,
)
from xai_ids.interfaces.model import Model as ModelABC
from xai_ids.registry import Registry


@pytest.mark.parametrize("abc", [Dataset, Model, Explainer, Attack, Metric])
def test_abstract_classes_cannot_instantiate(abc) -> None:
    with pytest.raises(TypeError):
        abc()  # type: ignore[abstract]


def test_databundle_validation() -> None:
    ok = DataBundle(
        X_train=np.zeros((4, 3)),
        y_train=np.zeros(4),
        X_test=np.zeros((2, 3)),
        y_test=np.zeros(2),
        X_attack=np.zeros((2, 3)),
        attack_indices=np.arange(2),
        feature_names=["a", "b", "c"],
    )
    assert ok.n_features == 3

    with pytest.raises(ValueError):
        DataBundle(
            X_train=np.zeros((4, 3)),
            y_train=np.zeros(4),
            X_test=np.zeros((2, 5)),  # mismatched feature count
            y_test=np.zeros(2),
            X_attack=np.zeros((2, 5)),
            attack_indices=np.arange(2),
            feature_names=["a", "b", "c"],
        )


def test_attack_result_retention_rate() -> None:
    res = AttackResult(
        X_adv=np.zeros((4, 2)),
        prediction_preserved=np.array([True, True, False, True]),
    )
    assert res.retention_rate == 0.75

    with pytest.raises(ValueError):
        AttackResult(
            X_adv=np.zeros((4, 2)),
            prediction_preserved=np.array([True, False]),  # wrong length
        )


class _DummyExplainer(Explainer):
    name = "dummy"

    def explain(self, model: Model, X: np.ndarray) -> np.ndarray:
        # deterministic attributions equal to the inputs
        return np.asarray(X, dtype=float)


def test_explainer_target_features() -> None:
    exp = _DummyExplainer()
    attributions = np.array([[0.1, 0.9, 0.2], [0.5, 0.4, 0.3]])
    top1 = exp.target_features(attributions, k=1)
    assert top1.tolist() == [[1], [0]]


def test_explainer_noise_floor_zero_for_deterministic() -> None:
    exp = _DummyExplainer()
    X = np.array([[1.0, 2.0, 3.0]])
    nf = exp.noise_floor(model=None, X=X, repeats=5)  # type: ignore[arg-type]
    assert np.allclose(nf, 0.0)


def test_registry() -> None:
    reg = Registry("thing")

    @reg.register("foo")
    class _Foo:
        pass

    assert "foo" in reg
    assert reg.get("foo") is _Foo
    assert reg.available() == ["foo"]

    with pytest.raises(KeyError):
        reg.get("missing")

    with pytest.raises(KeyError):

        @reg.register("foo")  # duplicate
        class _Foo2:
            pass


def test_model_abc_reexport_identity() -> None:
    # The package-level export and the module class are the same object.
    assert Model is ModelABC
