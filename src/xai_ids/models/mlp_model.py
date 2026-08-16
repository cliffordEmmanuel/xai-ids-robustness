"""Multi-Layer Perceptron IDS classifier (PyTorch).

Class imbalance is handled with a weighted ``BCEWithLogitsLoss`` (``pos_weight``
= neg/pos) rather than resampling. ``torch`` is imported lazily so the core CLI
works without it. ``predict_proba`` is the black-box surface used by the attacks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..interfaces.model import Model
from ..registry import MODELS

_log = logging.getLogger("xai_ids")

_DEFAULTS = {
    "hidden": (128, 64),
    "epochs": 20,
    "lr": 1e-3,
    "batch_size": 512,
    "seed": 42,
}


@MODELS.register("mlp")
class MLPModel(Model):
    name = "mlp"

    def __init__(self, class_weight: bool = True, params: dict | None = None):
        self.class_weight = class_weight
        self.params = {**_DEFAULTS, **(params or {})}
        self._model = None
        self._in_dim: int | None = None

    def _build(self, in_dim: int):
        from torch import nn

        layers, prev = [], in_dim
        for h in self.params["hidden"]:
            layers += [nn.Linear(prev, int(h)), nn.ReLU()]
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPModel:
        import torch
        from torch import nn

        torch.manual_seed(int(self.params["seed"]))
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
        self._in_dim = Xt.shape[1]
        self._model = self._build(self._in_dim)

        pos_weight = None
        if self.class_weight:
            pos, neg = float((yt == 1).sum()), float((yt == 0).sum())
            pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self._model.parameters(), lr=float(self.params["lr"]))

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=int(self.params["batch_size"]),
            shuffle=True,
        )
        epochs = int(self.params["epochs"])
        every = max(1, epochs // 10)
        self._model.train()
        for epoch in range(1, epochs + 1):
            total, n_batches = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self._model(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss.item())
                n_batches += 1
            if epoch % every == 0 or epoch == epochs:
                _log.info("  epoch %d/%d loss=%.4f", epoch, epochs, total / max(n_batches, 1))
        return self

    def _logits(self, X: np.ndarray):
        import torch

        self._model.eval()
        with torch.no_grad():
            return self._model(torch.tensor(np.asarray(X), dtype=torch.float32))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        p = torch.sigmoid(self._logits(X)).numpy().ravel()
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": self._model.state_dict(),
                "in_dim": self._in_dim,
                "params": self.params,
                "class_weight": self.class_weight,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> MLPModel:
        import torch

        ckpt = torch.load(path, weights_only=False)
        obj = cls(class_weight=ckpt["class_weight"], params=ckpt["params"])
        obj._in_dim = ckpt["in_dim"]
        obj._model = obj._build(obj._in_dim)
        obj._model.load_state_dict(ckpt["state"])
        return obj
