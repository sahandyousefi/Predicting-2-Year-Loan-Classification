import numpy as np
import pandas as pd

from credit_risk.explain.fairness import age_band, group_report


def test_age_band_edges():
    bands = age_band(pd.Series([24, 25, 61, 62, 80]))
    assert list(bands.astype(str)) == ["<25", "25-34", "55-61", "62+", "62+"]


def test_parity_when_scores_ignore_group():
    rng = np.random.default_rng(0)
    n = 6000
    groups = pd.Series(rng.choice(["a", "b"], n))
    p = rng.uniform(size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    rep = group_report(y, p, groups, threshold=0.5)
    assert rep["disparate_impact"].min() > 0.9
    assert not rep["four_fifths_flag"].any()


def test_flags_disadvantaged_group():
    rng = np.random.default_rng(1)
    n = 4000
    groups = pd.Series(np.where(np.arange(n) < n // 2, "a", "b"))
    p = rng.uniform(size=n) + np.where(groups == "b", 0.3, 0.0)
    y = (rng.uniform(size=n) < np.clip(p, 0, 1)).astype(int)
    rep = group_report(y, p, groups, threshold=0.5).set_index("group")
    assert rep.loc["b", "four_fifths_flag"]
    assert rep.loc["b", "equal_opportunity_diff"] < 0
