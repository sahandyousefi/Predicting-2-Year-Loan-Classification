import numpy as np
import pandas as pd
import pytest

from credit_risk.cli import main
from credit_risk.data.synthetic import SyntheticSpec, make_synthetic
from credit_risk.monitoring.drift import drift_report, psi, psi_status, score_psi


def test_psi_zero_for_same_sample():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=10_000)
    small = psi(ref, rng.normal(0.1, 1, 10_000))
    large = psi(ref, rng.normal(1.0, 1, 10_000))
    assert small < 0.1 < 0.25 < large


def test_discrete_feature_bins():
    ref = pd.Series([0] * 900 + [1] * 80 + [2] * 20)
    cur = pd.Series([0] * 600 + [1] * 300 + [2] * 100)
    assert psi_status(psi(ref, cur)) == "significant"


def test_report_flags_shifted_feature():
    ref = make_synthetic(SyntheticSpec(n=3000, seed=1))
    cur = make_synthetic(SyntheticSpec(n=3000, seed=2))
    cur["monthly_inc"] *= 1.8
    report = drift_report(ref, cur, ["age", "monthly_inc"]).set_index("feature")
    assert report.loc["monthly_inc", "status"] == "significant"
    assert report.loc["age", "status"] == "stable"
    assert score_psi(ref["age"], cur["age"])["status"] == "stable"


def test_drift_cli(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    main(["synthesize", str(a), "--seed", "1"])
    main(["synthesize", str(b), "--seed", "2"])
    assert main(["drift", str(a), str(b), "--fail-on-drift"]) == 0
