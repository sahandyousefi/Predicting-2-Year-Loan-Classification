import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk.evaluation.delong import auc_confidence_interval, delong_test


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    n = 1500
    y = rng.integers(0, 2, n)
    strong = y * 1.2 + rng.normal(size=n)
    weak = y * 0.4 + rng.normal(size=n)
    return y, strong, weak


def test_auc_matches_sklearn(sample):
    y, strong, weak = sample
    res = delong_test(y, strong, weak)
    assert res.auc_a == pytest.approx(roc_auc_score(y, strong))
    assert res.auc_b == pytest.approx(roc_auc_score(y, weak))


def test_detects_real_difference(sample):
    y, strong, weak = sample
    res = delong_test(y, strong, weak)
    assert res.p_value < 1e-6
    assert res.ci_low > 0


def test_identical_scores_not_significant(sample):
    y, strong, _ = sample
    rng = np.random.default_rng(1)
    jitter = strong + rng.normal(scale=1e-3, size=len(strong))
    assert delong_test(y, strong, jitter).p_value > 0.05


def test_null_rejection_rate_near_alpha():
    rng = np.random.default_rng(5)
    rejections = 0
    for _ in range(200):
        y = rng.integers(0, 2, 300)
        a = y * 0.8 + rng.normal(size=300)
        b = y * 0.8 + rng.normal(size=300)
        rejections += delong_test(y, a, b).p_value < 0.05
    assert rejections / 200 < 0.1


def test_auc_interval_contains_point(sample):
    y, strong, _ = sample
    auc, lo, hi = auc_confidence_interval(y, strong)
    assert lo < auc < hi
