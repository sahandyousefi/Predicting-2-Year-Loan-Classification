"""Stacked ensemble on out-of-fold predictions.

Each base model is fitted on :math:`K - 1` folds and predicts the held-out
fold, which gives an out-of-fold (OOF) prediction for every training row
that the model did not see (Wolpert, 1992). A logistic meta-learner on the
OOF log-odds,

.. math::

    \\operatorname{logit} \\hat p(x) = w_0 + \\sum_m w_m \\operatorname{logit}
    \\hat p_m(x), \\qquad w_m \\ge 0,

combines the models. The non-negativity constraint follows Breiman (1996):
it keeps the combination interpretable as a weighting and stops the
meta-learner from exploiting the strong correlation between base predictions
with offsetting weights. Because the meta-learner is fitted on the unweighted
log loss, it also recalibrates base models trained with class weights. Base
models are refitted on all training rows for prediction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, _EPS, 1 - _EPS))


def fit_nonnegative_logistic(
    Z: np.ndarray, y: np.ndarray, l2: float = 1e-3
) -> tuple[float, np.ndarray]:
    """Logistic regression with non-negative slopes and a free intercept.

    Parameters
    ----------
    Z : ndarray of shape (n, m)
        Inputs, here the base model log-odds.
    y : ndarray of {0, 1}
        Outcomes.
    l2 : float
        Ridge penalty on the slopes per observation, for numerical stability
        when base predictions are almost collinear.
    """
    n, m = Z.shape
    y = np.asarray(y, dtype=float)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] + Z @ theta[1:]
        p = expit(eta)
        loss = (
            np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.5 * l2 * theta[1:] @ theta[1:]
        )
        r = (p - y) / n
        grad = np.concatenate([[r.sum()], Z.T @ r + l2 * theta[1:]])
        return float(loss), grad

    theta0 = np.concatenate([[0.0], np.full(m, 1.0 / m)])
    bounds = [(None, None)] + [(0.0, None)] * m
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
    return float(res.x[0]), res.x[1:]


class StackedEnsemble(ClassifierMixin, BaseEstimator):
    """Non-negative logistic stack of registry models.

    Parameters
    ----------
    base_models : sequence of str
        Registry names of the base learners.
    n_splits : int
        Folds used to produce out-of-fold predictions.
    random_state : int
        Seed for the fold assignment.
    """

    def __init__(
        self,
        base_models: Sequence[str] = ("logistic", "scorecard", "hist_gbm"),
        n_splits: int = 5,
        random_state: int = 0,
    ) -> None:
        self.base_models = base_models
        self.n_splits = n_splits
        self.random_state = random_state

    def _build(self, name: str):
        from credit_risk.models.registry import build_model

        return build_model(name)

    def fit(self, X: pd.DataFrame, y) -> StackedEnsemble:
        names = list(self.base_models)
        if len(names) < 2:
            raise ValueError("a stack needs at least two base models")
        if "stacked" in names:
            raise ValueError("a stack cannot contain itself")
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        folds = StratifiedKFold(
            self.n_splits, shuffle=True, random_state=self.random_state
        )
        oof = np.zeros((len(X), len(names)))
        for train_idx, test_idx in folds.split(X, y):
            for j, name in enumerate(names):
                model = self._build(name).fit(X.iloc[train_idx], y[train_idx])
                oof[test_idx, j] = model.predict_proba(X.iloc[test_idx])[:, 1]
        self.oof_ = oof
        self.oof_auc_ = {
            name: float(roc_auc_score(y, oof[:, j])) for j, name in enumerate(names)
        }
        self.intercept_, self.coef_ = fit_nonnegative_logistic(_logit(oof), y)
        # The meta-learner sees the same OOF rows, but with a handful of
        # parameters the optimism of this figure is negligible.
        self.oof_auc_["stacked"] = float(
            roc_auc_score(y, self.intercept_ + _logit(oof) @ self.coef_)
        )
        self.estimators_ = [self._build(name).fit(X, y) for name in names]
        return self

    @property
    def weights(self) -> dict[str, float]:
        """Meta-learner weight of each base model."""
        return {name: float(w) for name, w in zip(self.base_models, self.coef_)}

    def base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.estimators_])

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept_ + _logit(self.base_predictions(X)) @ self.coef_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = expit(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
