import numpy as np
import pytest

from credit_risk.config import CostConfig
from credit_risk.evaluation.thresholds import (
    breakeven_pd,
    optimal_threshold,
    profit_curve,
)


def test_breakeven_formula():
    cost = CostConfig(lgd=0.45, ead=1.0, margin=0.05)
    assert breakeven_pd(cost) == pytest.approx(0.1)


def test_optimal_threshold_near_breakeven_for_calibrated_pd():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 6, 50_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    cost = CostConfig(lgd=0.45, ead=1.0, margin=0.05)
    res = optimal_threshold(y, p, cost)
    assert res.threshold == pytest.approx(breakeven_pd(cost), abs=0.02)
    assert res.profit_per_applicant > 0


def test_curve_endpoints():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.6])
    curve = profit_curve(y, p, CostConfig())
    assert curve["approval_rate"].iloc[0] == 0
    assert curve["profit_per_applicant"].iloc[0] == 0
    assert curve["approval_rate"].iloc[-1] == 1
