import numpy as np
import pandas as pd

from credit_risk.data.schema import FEATURES, TARGET
from credit_risk.features.woe import WoEBinner, iv_strength


def test_woe_transform_shape_and_finite(credit_df):
    X, y = credit_df[FEATURES], credit_df[TARGET]
    binner = WoEBinner().fit(X, y)
    Z = binner.transform(X)
    assert Z.shape == X.shape
    assert np.isfinite(Z.to_numpy()).all()


def test_monotonic_event_rate(credit_df):
    binner = WoEBinner(max_bins=10, monotonic=True).fit(
        credit_df[["age"]], credit_df[TARGET]
    )
    rates = binner.bin_table_["event_rate"].to_numpy()
    diffs = np.diff(rates)
    assert (diffs <= 1e-12).all() or (diffs >= -1e-12).all()


def test_iv_ranks_signal_above_noise():
    rng = np.random.default_rng(0)
    n = 3000
    signal = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-2 * signal))).astype(int)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    iv = WoEBinner().fit(X, y).iv_
    assert iv["signal"] > 0.3
    assert iv["noise"] < 0.05


def test_small_bins_are_merged(credit_df):
    binner = WoEBinner(max_bins=20, min_bin_share=0.1, monotonic=False).fit(
        credit_df[["debt_ratio"]], credit_df[TARGET]
    )
    assert binner.bin_table_["count"].min() >= 0.1 * len(credit_df)


def test_iv_strength_labels():
    assert iv_strength(0.01) == "not predictive"
    assert iv_strength(0.5) == "strong"
