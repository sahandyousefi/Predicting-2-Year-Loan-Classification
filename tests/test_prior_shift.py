import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk.evaluation.calibration import expected_calibration_error
from credit_risk.models.prior_shift import (
    adjust_to_prior,
    calibration_in_the_large,
    estimate_prior_em,
    weighted_prior,
)


def _shifted_sample(target_rate, n=20000, seed=0):
    """Calibrated PDs at a 0.5 prior, then defaulters subsampled."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.5, n)
    z = rng.normal(0, 1, n) + 1.5 * y
    # Exact posterior of the two-Gaussian model at prior 0.5.
    p = 1 / (1 + np.exp(-(1.5 * z - 1.125)))
    keep_bad = target_rate / (1 - target_rate)
    keep = (y == 0) | (rng.uniform(size=n) < keep_bad)
    return y[keep], p[keep]


def test_adjustment_is_identity_at_equal_rates():
    p = np.array([0.01, 0.3, 0.9])
    assert adjust_to_prior(p, 0.2, 0.2) == pytest.approx(p)


def test_adjustment_is_invertible_and_rank_preserving():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 500)
    y = rng.binomial(1, p)
    q = adjust_to_prior(p, 0.5, 0.05)
    assert adjust_to_prior(q, 0.05, 0.5) == pytest.approx(p)
    assert roc_auc_score(y, q) == pytest.approx(roc_auc_score(y, p))
    assert (q < p).all()


def test_adjustment_recalibrates_to_target_rate():
    y, p = _shifted_sample(0.08)
    q = adjust_to_prior(p, 0.5, y.mean())
    assert q.mean() == pytest.approx(y.mean(), abs=0.01)
    assert expected_calibration_error(y, q) < 0.01
    assert expected_calibration_error(y, p) > 0.2


def test_em_recovers_unknown_prior():
    y, p = _shifted_sample(0.1, seed=2)
    est = estimate_prior_em(p, 0.5)
    assert est.converged
    assert est.target_rate == pytest.approx(y.mean(), abs=0.015)
    assert est.adjusted.mean() == pytest.approx(est.target_rate, abs=1e-4)


def test_em_stays_at_source_without_shift():
    y, p = _shifted_sample(0.5, seed=3)
    assert estimate_prior_em(p, 0.5).target_rate == pytest.approx(0.5, abs=0.02)


def test_weighted_prior():
    rate = 0.07
    balanced = (0.5 / (1 - rate), 0.5 / rate)
    assert weighted_prior(rate, *balanced) == pytest.approx(0.5)
    assert weighted_prior(rate, 1.0, 1.0) == pytest.approx(rate)
    with pytest.raises(ValueError):
        weighted_prior(0.0, 1.0, 1.0)


def test_calibration_in_the_large():
    y, p = _shifted_sample(0.08, seed=4)
    raw = calibration_in_the_large(y, p)
    fixed = calibration_in_the_large(y, adjust_to_prior(p, 0.5, y.mean()))
    assert raw["p_value"] < 1e-6 and raw["z"] < 0
    assert abs(fixed["z"]) < abs(raw["z"])
    assert fixed["mean_pd"] == pytest.approx(fixed["observed_rate"], abs=0.01)
