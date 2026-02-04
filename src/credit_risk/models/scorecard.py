from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from credit_risk.features.woe import WoEBinner
from credit_risk.models.linear import make_logistic


class Scorecard(BaseEstimator, ClassifierMixin):
    """Logistic regression on WoE features, scaled to scorecard points.

    The score is linear in the log-odds of being a good borrower::

        score  = offset + factor * ln(odds_good)
        factor = pdo / ln(2)
        offset = base_score - factor * ln(base_odds)

    so that ``base_score`` corresponds to good:bad odds of ``base_odds`` and
    every ``pdo`` points double the odds. Each feature bin receives integer
    points, and the intercept is spread evenly across features.

    Class weights are not used, so predicted probabilities stay on the scale of
    the training default rate.
    """

    def __init__(
        self,
        base_score: float = 600.0,
        base_odds: float = 50.0,
        pdo: float = 20.0,
        max_bins: int = 8,
        min_bin_share: float = 0.05,
        C: float = 1.0,
    ) -> None:
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.max_bins = max_bins
        self.min_bin_share = min_bin_share
        self.C = C

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def fit(self, X: pd.DataFrame, y) -> Scorecard:
        self.binner_ = WoEBinner(
            max_bins=self.max_bins, min_bin_share=self.min_bin_share
        ).fit(X, y)
        Z = self.binner_.transform(X)
        self.model_ = make_logistic(C=self.C, l1_ratio=0.0, class_weight=None)
        self.model_.fit(Z, y)
        self.classes_ = self.model_.classes_
        self.points_table_ = self._points_table()
        return self

    def _points_table(self) -> pd.DataFrame:
        table = self.binner_.bin_table_[["feature", "bin", "woe"]].copy()
        coefs = dict(zip(self.binner_.feature_names_in_, self.model_.coef_[0]))
        n = len(coefs)
        b0 = self.model_.intercept_[0]
        # Model predicts the event (bad), so good log-odds is the negative logit.
        raw = -(table["feature"].map(coefs) * table["woe"] + b0 / n)
        table["points"] = np.round(raw * self.factor + self.offset / n).astype(int)
        return table

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.binner_.transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.binner_.transform(X))

    def score(self, X: pd.DataFrame, y=None) -> np.ndarray:  # noqa: ARG002
        """Total points per applicant (higher means lower risk)."""
        total = np.zeros(len(X))
        for col in self.binner_.feature_names_in_:
            pts = self.points_table_.loc[
                self.points_table_["feature"] == col, "points"
            ].to_numpy()
            idx = np.searchsorted(
                self.binner_.edges_[col], X[col].to_numpy(dtype=float), side="right"
            )
            total += pts[idx]
        return total

    def score_to_pd(self, score: np.ndarray) -> np.ndarray:
        log_odds_good = (np.asarray(score) - self.offset) / self.factor
        return 1.0 / (1.0 + np.exp(log_odds_good))
