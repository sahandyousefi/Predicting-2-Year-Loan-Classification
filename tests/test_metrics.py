import numpy as np
import pytest

from credit_risk.evaluation.metrics import (
    credit_metrics,
    gini,
    ks_statistic,
    lift_gain_table,
)


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    score = y * 1.0 + rng.normal(size=2000)
    return y, score


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert ks_statistic(y, s) == 1.0
    assert gini(y, s) == 1.0


def test_ks_matches_scipy(scored):
    from scipy.stats import ks_2samp

    y, s = scored
    expected = ks_2samp(s[y == 1], s[y == 0]).statistic
    assert ks_statistic(y, s) == pytest.approx(expected, abs=1e-9)


def test_gain_reaches_one_and_lift_decreases(scored):
    y, s = scored
    table = lift_gain_table(y, s)
    assert len(table) == 10
    assert table["cum_gain"].iloc[-1] == pytest.approx(1.0)
    assert table["lift"].iloc[0] > 1 > table["lift"].iloc[-1]
    assert table["count"].sum() == len(y)


def test_credit_metrics_keys(scored):
    assert set(credit_metrics(*scored)) == {"roc_auc", "gini", "ks", "pr_auc"}
