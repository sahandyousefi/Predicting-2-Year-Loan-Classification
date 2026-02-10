"""Synthetic monthly performance histories for time-to-default modeling.

The benchmark records only whether a borrower defaulted within two years. A
survival model needs the month of default and the month an account left
observation for other reasons (prepayment, closure, end of data). This module
simulates both from a known discrete-time hazard, so that fitted hazard models
can be checked against the true cumulative PD of every borrower.

The monthly default hazard is

.. math::

    h(t \\mid x) = \\operatorname{expit}\\{\\alpha + \\log b(t) + \\eta(x)\\},
    \\qquad b(t) = (t / t_p) \\exp(1 - t / t_p),

a hump-shaped baseline that peaks at month :math:`t_p`, with :math:`\\eta` a
linear risk index of the standardized attributes. Independent censoring
arrives with a constant monthly hazard and every account is censored
administratively at ``max_months``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from credit_risk.data.schema import FEATURES
from credit_risk.data.synthetic import SyntheticSpec, make_synthetic
from credit_risk.features.engineering import add_engineered_features

DURATION = "duration"
EVENT = "event"
_SURVIVAL_STREAM = 303

RISK_WEIGHTS: dict[str, float] = {
    "late_severity": 0.8,
    "rev_util": 0.6,
    "log_debt_ratio": 0.2,
    "age": -0.4,
    "log_monthly_inc": -0.3,
}


@dataclass(frozen=True)
class SurvivalSpec:
    """Parameters of the simulated performance window.

    Attributes
    ----------
    n : int
        Number of accounts.
    max_months : int
        Length of the observation window.
    pd_24 : float
        Target mean probability of default within 24 months, before censoring.
    peak_month : float
        Month at which the baseline hazard peaks.
    censor_rate : float
        Constant monthly hazard of leaving observation without default.
    seed : int
        Random seed.
    """

    n: int = 5000
    max_months: int = 36
    pd_24: float = 0.15
    peak_month: float = 9.0
    censor_rate: float = 0.015
    seed: int = 0


@dataclass
class SurvivalSample:
    """Simulated accounts and the hazard that generated them.

    ``frame`` holds the benchmark attributes plus ``duration`` (last month
    observed, 1 to ``max_months``) and ``event`` (1 if that month is a
    default). ``hazard`` is the true ``(n, max_months)`` monthly hazard.
    """

    frame: pd.DataFrame
    hazard: np.ndarray

    def true_cumulative_pd(self, horizon: int) -> np.ndarray:
        """Probability of default by month ``horizon`` without censoring."""
        return 1.0 - np.prod(1.0 - self.hazard[:, :horizon], axis=1)


def risk_index(df: pd.DataFrame) -> np.ndarray:
    """Linear risk index of standardized attributes used by the simulator."""
    fe = add_engineered_features(df)
    eta = np.zeros(len(df))
    for col, w in RISK_WEIGHTS.items():
        x = fe[col].to_numpy(dtype=float)
        eta += w * (x - x.mean()) / (x.std() + 1e-12)
    return eta


def baseline_shape(max_months: int, peak_month: float) -> np.ndarray:
    t = np.arange(1, max_months + 1, dtype=float)
    return (t / peak_month) * np.exp(1.0 - t / peak_month)


def simulate_survival(spec: SurvivalSpec | None = None) -> SurvivalSample:
    """Simulate accounts with monthly default and censoring times."""
    spec = spec or SurvivalSpec()
    if spec.max_months < 24:
        raise ValueError("max_months must be at least 24")
    attrs = make_synthetic(SyntheticSpec(n=spec.n, seed=spec.seed))[FEATURES]
    eta = risk_index(attrs)
    log_b = np.log(baseline_shape(spec.max_months, spec.peak_month))

    def hazard(alpha: float) -> np.ndarray:
        return expit(alpha + log_b[None, :] + eta[:, None])

    def pd_24(alpha: float) -> float:
        return float(1.0 - np.prod(1.0 - hazard(alpha)[:, :24], axis=1).mean())

    alpha = brentq(lambda a: pd_24(a) - spec.pd_24, -20, 5)
    h = hazard(alpha)

    rng = np.random.default_rng([spec.seed, _SURVIVAL_STREAM])
    hit = rng.uniform(size=h.shape) < h
    default_month = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, np.inf)
    censor_month = (
        rng.geometric(spec.censor_rate, size=spec.n).astype(float)
        if spec.censor_rate > 0
        else np.full(spec.n, np.inf)
    )
    censor_month = np.minimum(censor_month, spec.max_months)
    event = default_month <= censor_month
    duration = np.where(event, default_month, censor_month).astype(int)

    frame = attrs.reset_index(drop=True).copy()
    frame[DURATION] = duration
    frame[EVENT] = event.astype(int)
    return SurvivalSample(frame, h)


def binary_label(
    frame: pd.DataFrame, horizon: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """Default-within-horizon label and a mask of accounts where it is known.

    The label is known when the account defaulted by ``horizon`` or was still
    observed at ``horizon``.
    """
    d = frame[DURATION].to_numpy()
    e = frame[EVENT].to_numpy().astype(bool)
    y = (e & (d <= horizon)).astype(int)
    known = (e & (d <= horizon)) | (d >= horizon)
    return y, known
