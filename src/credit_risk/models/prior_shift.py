"""Recalibration of PDs to a different portfolio default rate.

The benchmark sample has a default rate far above that of a consumer
portfolio, and class weighting moves the effective training prior further.
Under prior shift, where the class-conditional distributions of the inputs
are unchanged and only the share of defaulters differs, the posterior
probability transforms exactly by a shift of the log-odds (Saerens et al.,
2002). When the target default rate is unknown it can be estimated from
unlabeled target scores with the EM procedure of the same paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

_EPS = 1e-12


def _check_rate(rate: float, name: str) -> None:
    if not 0 < rate < 1:
        raise ValueError(f"{name} must be in (0, 1), got {rate}")


def adjust_to_prior(pd_hat, source_rate: float, target_rate: float) -> np.ndarray:
    """PD under the target default rate.

    ``logit p' = logit p + logit(target_rate) - logit(source_rate)``.
    Rank order, and therefore AUC, Gini and KS, are unchanged.
    """
    _check_rate(source_rate, "source_rate")
    _check_rate(target_rate, "target_rate")
    p = np.clip(np.asarray(pd_hat, dtype=float), _EPS, 1 - _EPS)
    return expit(logit(p) + logit(target_rate) - logit(source_rate))


def weighted_prior(base_rate: float, weight_0: float, weight_1: float) -> float:
    """Effective default rate seen by a model fitted with class weights.

    With ``class_weight="balanced"`` the weights are inversely proportional to
    the class shares and the effective rate is 0.5.
    """
    _check_rate(base_rate, "base_rate")
    num = weight_1 * base_rate
    return float(num / (num + weight_0 * (1 - base_rate)))


@dataclass
class PriorEstimate:
    target_rate: float
    adjusted: np.ndarray
    n_iter: int
    converged: bool


def estimate_prior_em(
    pd_hat,
    source_rate: float,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> PriorEstimate:
    """Estimate the default rate of an unlabeled sample by EM.

    Starting from the source rate, each step adjusts the PDs to the current
    rate (E-step) and sets the rate to their mean (M-step). The fixed point is
    the maximum likelihood estimate of the target prior, assuming the source
    PDs are calibrated and only the prior has changed.
    """
    _check_rate(source_rate, "source_rate")
    rate = source_rate
    for i in range(1, max_iter + 1):
        adjusted = adjust_to_prior(pd_hat, source_rate, rate)
        new_rate = float(np.clip(adjusted.mean(), _EPS, 1 - _EPS))
        if abs(new_rate - rate) < tol:
            return PriorEstimate(new_rate, adjusted, i, True)
        rate = new_rate
    return PriorEstimate(rate, adjust_to_prior(pd_hat, source_rate, rate), i, False)


def calibration_in_the_large(y_true, pd_hat) -> dict[str, float]:
    """Observed against expected defaults with a normal-approximation test.

    The statistic ``z = (O - E) / sqrt(sum p_i (1 - p_i))`` treats defaults as
    independent Bernoulli trials; correlated defaults make the test
    anti-conservative.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pd_hat, dtype=float)
    observed, expected = y.sum(), p.sum()
    z = (observed - expected) / np.sqrt(max((p * (1 - p)).sum(), _EPS))
    return {
        "observed_rate": float(y.mean()),
        "mean_pd": float(p.mean()),
        "z": float(z),
        "p_value": float(2 * norm.sf(abs(z))),
    }
