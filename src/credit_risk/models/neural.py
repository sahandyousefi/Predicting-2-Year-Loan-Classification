"""Multilayer perceptron baseline for tabular credit data.

The network is a scikit-learn compatible classifier built on PyTorch, so it
runs through the same comparison, calibration and persistence code as the
other models. Numerical inputs are standardized and clipped, or encoded with
the piecewise linear encoding of Gorishniy et al. (2022), which gives an MLP
a step-like view of each attribute similar to what trees see. Training uses
AdamW with early stopping on the log-loss of an internal validation split.

PyTorch is an optional dependency: ``pip install -e ".[nn]"``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PiecewiseLinearEncoder(TransformerMixin, BaseEstimator):
    """Piecewise linear encoding of numerical features on quantile bins.

    For bin edges :math:`b_0 < \\dots < b_T` of feature :math:`x`, component
    :math:`t` equals 0 below :math:`b_{t-1}`, 1 above :math:`b_t` and
    :math:`(x - b_{t-1}) / (b_t - b_{t-1})` in between. The encoding is
    lossless up to the outer edges and monotone in :math:`x`.

    Parameters
    ----------
    n_bins : int
        Maximum number of quantile bins per feature. Duplicate edges, common
        for counts, are removed, so discrete features get fewer bins.
    """

    def __init__(self, n_bins: int = 16) -> None:
        self.n_bins = n_bins

    def fit(self, X, y=None) -> PiecewiseLinearEncoder:
        X = np.asarray(X, dtype=float)
        q = np.linspace(0, 1, self.n_bins + 1)
        self.edges_ = []
        for j in range(X.shape[1]):
            edges = np.unique(np.quantile(X[:, j], q))
            if len(edges) < 2:
                edges = np.array([edges[0], edges[0] + 1.0])
            self.edges_.append(edges)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("unexpected number of features")
        blocks = []
        for j, edges in enumerate(self.edges_):
            lo, hi = edges[:-1], edges[1:]
            frac = (X[:, [j]] - lo[None, :]) / (hi - lo)[None, :]
            blocks.append(np.clip(frac, 0.0, 1.0))
        return np.hstack(blocks)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("TabularMLP needs PyTorch: pip install -e .[nn]") from exc
    return torch


class TabularMLP(ClassifierMixin, BaseEstimator):
    """Feed-forward network for binary default prediction.

    Parameters
    ----------
    hidden : sequence of int
        Width of each hidden layer.
    dropout : float
        Dropout rate after each hidden layer.
    encoding : {"standard", "ple"}
        Numerical input encoding: standardization clipped to five standard
        deviations, or piecewise linear encoding on quantile bins.
    n_bins : int
        Bins per feature for the piecewise linear encoding.
    lr, weight_decay : float
        AdamW learning rate and decoupled weight decay.
    batch_size, max_epochs, patience : int
        Mini-batch size, epoch limit and early stopping patience in epochs.
    validation_fraction : float
        Share of the training data held out for early stopping.
    random_state : int
        Seed for the split, initialization and batch order.
    """

    def __init__(
        self,
        hidden: Sequence[int] = (64, 32),
        dropout: float = 0.1,
        encoding: str = "ple",
        n_bins: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        validation_fraction: float = 0.15,
        random_state: int = 0,
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.encoding = encoding
        self.n_bins = n_bins
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _encoder(self):
        if self.encoding == "ple":
            return PiecewiseLinearEncoder(self.n_bins)
        if self.encoding == "standard":
            return StandardScaler()
        raise ValueError("encoding must be 'standard' or 'ple'")

    def _encode(self, X) -> np.ndarray:
        Z = self.encoder_.transform(np.asarray(X, dtype=float))
        if self.encoding == "standard":
            Z = np.clip(Z, -5.0, 5.0)
        return Z.astype(np.float32)

    def _network(self, n_in: int):
        nn = _torch().nn
        layers: list = []
        for width in self.hidden:
            layers += [nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(self.dropout)]
            n_in = width
        layers.append(nn.Linear(n_in, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y) -> TabularMLP:
        torch = _torch()
        torch.manual_seed(self.random_state)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        X_tr, X_va, y_tr, y_va = train_test_split(
            np.asarray(X, dtype=float),
            y,
            test_size=self.validation_fraction,
            stratify=y,
            random_state=self.random_state,
        )
        self.encoder_ = self._encoder().fit(X_tr)
        Z_tr = torch.from_numpy(self._encode(X_tr))
        Z_va = torch.from_numpy(self._encode(X_va))
        t_tr, t_va = torch.from_numpy(y_tr), torch.from_numpy(y_va)

        net = self._network(Z_tr.shape[1])
        opt = torch.optim.AdamW(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.random_state)
        best, best_state, wait = np.inf, None, 0
        self.history_: list[tuple[float, float]] = []
        for epoch in range(self.max_epochs):
            net.train()
            order = torch.randperm(len(Z_tr), generator=gen)
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Z_tr[idx]).squeeze(1), t_tr[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            net.eval()
            with torch.no_grad():
                val = float(loss_fn(net(Z_va).squeeze(1), t_va))
            self.history_.append((total / len(Z_tr), val))
            if val < best - 1e-5:
                best, wait, self.best_epoch_ = val, 0, epoch + 1
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.best_val_loss_ = best
        return self

    def decision_function(self, X) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(self._encode(X))).squeeze(1)
        return logits.numpy().astype(float)

    def predict_proba(self, X) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getstate__(self) -> dict:
        # Store weights as NumPy arrays so joblib bundles do not pickle modules.
        state = self.__dict__.copy()
        if "net_" in state:
            state["net_weights_"] = {
                k: v.numpy().copy() for k, v in state.pop("net_").state_dict().items()
            }
        return state

    def __setstate__(self, state: dict) -> None:
        weights = state.pop("net_weights_", None)
        self.__dict__.update(state)
        if weights is not None:
            torch = _torch()
            net = self._network(self._encode_width())
            net.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
            net.eval()
            self.net_ = net

    def _encode_width(self) -> int:
        if self.encoding == "ple":
            return int(sum(len(e) - 1 for e in self.encoder_.edges_))
        return int(self.encoder_.n_features_in_)
