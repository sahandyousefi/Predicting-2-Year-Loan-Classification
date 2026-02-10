"""Discrete-time survival model for time to default.

Each account contributes one row per month it was observed (person-period
format). A logistic regression on those rows estimates the monthly hazard

.. math::

    h(t \\mid x) = \\Pr(T = t \\mid T \\ge t, x)
    = \\operatorname{expit}\\{\\alpha_{k(t)} + x^\\top \\beta\\},

with a piecewise-constant baseline :math:`\\alpha_k` on the intervals given by
``time_bins``. The likelihood of the person-period logistic regression equals
the discrete-time survival likelihood with right censoring (Allison, 1982), so
accounts that leave observation early contribute the months they were seen
instead of being dropped or counted as good. The cumulative PD at any horizon
follows from :math:`S(t \\mid x) = \\prod_{s \\le t} \\{1 - h(s \\mid x)\\}`,
which gives a 12-month and a lifetime PD from one model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def person_period(
    X: pd.DataFrame | np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand accounts into one row per observed month.

    Returns
    -------
    rows : ndarray
        Index of the source account for each person-period row.
    period : ndarray
        Month number, starting at 1.
    y : ndarray
        1 in the month of default, 0 otherwise.
    """
    duration = np.asarray(duration, dtype=int)
    event = np.asarray(event, dtype=int)
    if np.any(duration < 1):
        raise ValueError("durations must be at least 1")
    if len(duration) != len(X) or len(event) != len(X):
        raise ValueError("X, duration and event must have the same length")
    rows = np.repeat(np.arange(len(duration)), duration)
    ends = np.cumsum(duration)
    period = np.arange(len(rows)) - np.repeat(ends - duration, duration) + 1
    y = np.zeros(len(rows), dtype=int)
    y[ends - 1] = event
    return rows, period, y


class DiscreteTimeHazard:
    """Logistic discrete-time hazard model with a piecewise-constant baseline.

    Parameters
    ----------
    time_bins : sequence of int
        Right edges of the baseline intervals in months. The last edge is the
        longest horizon the model can predict.
    C : float
        Inverse L2 penalty on the attribute coefficients.
    """

    def __init__(
        self, time_bins: Sequence[int] = (3, 6, 9, 12, 18, 24, 36), C: float = 1.0
    ) -> None:
        edges = list(time_bins)
        if edges != sorted(set(edges)) or edges[0] < 1:
            raise ValueError("time_bins must be strictly increasing positive months")
        self.time_bins = tuple(edges)
        self.C = C

    @property
    def max_time(self) -> int:
        return self.time_bins[-1]

    def _interval(self, period: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.time_bins, period, side="left")

    def _design(self, Z: np.ndarray, period: np.ndarray) -> np.ndarray:
        # Treatment coding: the intercept is the first interval's log-odds and
        # the dummies are contrasts, so the L2 penalty barely touches them.
        basis = np.eye(len(self.time_bins))[self._interval(period)][:, 1:]
        return np.hstack([basis, Z])

    def fit(
        self, X: pd.DataFrame | np.ndarray, duration: np.ndarray, event: np.ndarray
    ) -> DiscreteTimeHazard:
        duration = np.asarray(duration, dtype=int)
        event = np.asarray(event, dtype=int) * (duration <= self.max_time)
        duration = np.minimum(duration, self.max_time)
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        rows, period, y = person_period(Z, duration, event)
        self.model_ = LogisticRegression(C=self.C, max_iter=5000).fit(
            self._design(Z[rows], period), y
        )
        k = len(self.time_bins)
        coef = self.model_.coef_.ravel()
        intercept = float(self.model_.intercept_[0])
        self.baseline_ = intercept + np.concatenate([[0.0], coef[: k - 1]])
        self.coef_ = coef[k - 1 :]
        self.n_person_periods_ = len(y)
        return self

    def hazard(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Monthly hazard, shape ``(n, max_time)``."""
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        months = np.arange(1, self.max_time + 1)
        alpha = self.baseline_[self._interval(months)]
        return 1.0 / (1.0 + np.exp(-(alpha[None, :] + (Z @ self.coef_)[:, None])))

    def survival(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Survival function :math:`S(t \\mid x)` for months 1 to ``max_time``."""
        return np.cumprod(1.0 - self.hazard(X), axis=1)

    def cumulative_pd(self, X: pd.DataFrame | np.ndarray, horizon: int) -> np.ndarray:
        """Probability of default by month ``horizon``."""
        if not 1 <= horizon <= self.max_time:
            raise ValueError(f"horizon must be between 1 and {self.max_time}")
        return 1.0 - self.survival(X)[:, horizon - 1]


def kaplan_meier(duration: np.ndarray, event: np.ndarray, max_time: int) -> np.ndarray:
    """Kaplan-Meier survival estimate at months 1 to ``max_time``."""
    duration = np.asarray(duration, dtype=int)
    event = np.asarray(event, dtype=int)
    defaults = np.bincount(duration[event == 1], minlength=max_time + 2)
    exits = np.bincount(duration, minlength=max_time + 2)
    at_risk = len(duration) - np.concatenate([[0], np.cumsum(exits)[:-1]])
    t = np.arange(1, max_time + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(at_risk[t] > 0, defaults[t] / at_risk[t], 0.0)
    return np.cumprod(1.0 - h)


def concordance_index(
    duration: np.ndarray, event: np.ndarray, risk: np.ndarray, block: int = 2048
) -> float:
    """Harrell's C: share of comparable pairs ordered correctly by ``risk``.

    A pair is comparable when the account with the shorter duration defaulted.
    Ties in risk count one half.
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    concordant = 0.0
    comparable = 0.0
    idx = np.flatnonzero(event)
    for start in range(0, len(idx), block):
        i = idx[start : start + block]
        later = duration[None, :] > duration[i, None]
        diff = risk[i, None] - risk[None, :]
        concordant += float(
            ((diff > 0) & later).sum() + 0.5 * ((diff == 0) & later).sum()
        )
        comparable += float(later.sum())
    if comparable == 0:
        raise ValueError("no comparable pairs")
    return concordant / comparable


def survival_study(
    frame: pd.DataFrame,
    true_pd: dict[int, np.ndarray],
    horizon: int = 24,
    test_size: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare the hazard model with two binary treatments of censoring.

    The binary models predict default within ``horizon``. *Drop censored*
    keeps only accounts whose label is known; *censored as good* counts every
    account censored before ``horizon`` as non-default. Each model is scored
    on a holdout against the true cumulative PD of the simulator.

    Parameters
    ----------
    frame : DataFrame
        Output of :func:`credit_risk.data.survival.simulate_survival`.
    true_pd : dict
        True cumulative PD by horizon, aligned with ``frame``. Must contain
        ``horizon`` and may contain other horizons, which only the hazard
        model can predict.
    """
    from sklearn.model_selection import train_test_split

    from credit_risk.data.survival import DURATION, EVENT, binary_label
    from credit_risk.features.engineering import (
        FEATURE_COLUMNS,
        add_engineered_features,
    )

    X = add_engineered_features(frame)[FEATURE_COLUMNS].to_numpy(dtype=float)
    idx_tr, idx_te = train_test_split(
        np.arange(len(frame)), test_size=test_size, random_state=seed
    )
    d = frame[DURATION].to_numpy()
    e = frame[EVENT].to_numpy()
    y, known = binary_label(frame, horizon)

    def logistic(rows: np.ndarray):
        scaler = StandardScaler().fit(X[rows])
        clf = LogisticRegression(max_iter=5000).fit(scaler.transform(X[rows]), y[rows])
        return lambda Z, h: clf.predict_proba(scaler.transform(Z))[:, 1]

    hazard = DiscreteTimeHazard().fit(X[idx_tr], d[idx_tr], e[idx_tr])
    drop_rows = idx_tr[known[idx_tr]]
    # (predictor, training rows, horizons it can predict)
    candidates = {
        "Discrete-time hazard": (hazard.cumulative_pd, len(idx_tr), sorted(true_pd)),
        "Binary, drop censored": (logistic(drop_rows), len(drop_rows), [horizon]),
        "Binary, censored as good": (logistic(idx_tr), len(idx_tr), [horizon]),
    }
    rows = []
    for name, (predict, n_train, horizons) in candidates.items():
        for h in horizons:
            p = predict(X[idx_te], h)
            t = true_pd[h][idx_te]
            rows.append(
                {
                    "model": name,
                    "horizon": h,
                    "train_accounts": n_train,
                    "mean_pd": float(p.mean()),
                    "true_mean_pd": float(t.mean()),
                    "mae_vs_true": float(np.abs(p - t).mean()),
                    "c_index": concordance_index(d[idx_te], e[idx_te], p),
                }
            )
    return pd.DataFrame(rows)
