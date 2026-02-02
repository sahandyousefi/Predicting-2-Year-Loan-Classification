import numpy as np
import pytest

from credit_risk.data.schema import TARGET, validate_frame
from credit_risk.data.synthetic import SyntheticSpec, make_synthetic


def test_synthetic_passes_validation(credit_df):
    assert validate_frame(credit_df).ok


@pytest.mark.parametrize("rate", [0.1, 0.5])
def test_base_rate_is_close(rate):
    df = make_synthetic(SyntheticSpec(n=4000, base_rate=rate, seed=1))
    assert abs(df[TARGET].mean() - rate) < 0.03


def test_seed_reproducible():
    a = make_synthetic(SyntheticSpec(n=50, seed=11))
    b = make_synthetic(SyntheticSpec(n=50, seed=11))
    assert a.equals(b)


def test_late_counts_are_correlated(credit_df):
    corr = credit_df[["late_30_59", "late_60_89", "late_90"]].corr().to_numpy()
    assert corr[np.triu_indices(3, 1)].min() > 0.2
