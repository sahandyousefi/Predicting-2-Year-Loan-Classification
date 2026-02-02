import pytest

from credit_risk.data.synthetic import SyntheticSpec, make_synthetic


@pytest.fixture(scope="session")
def credit_df():
    return make_synthetic(SyntheticSpec(n=1500, seed=7))


@pytest.fixture
def small_df():
    return make_synthetic(SyntheticSpec(n=200, seed=3))
