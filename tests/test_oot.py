import pandas as pd
import pytest

from credit_risk.cli import main
from credit_risk.data.vintage import VINTAGE_COLUMN, add_vintage, out_of_time_split
from credit_risk.evaluation.oot import oot_validation


def test_vintages_are_balanced_and_ordered(credit_df):
    df = add_vintage(credit_df, start="2020-01", n_months=12)
    counts = df[VINTAGE_COLUMN].value_counts()
    assert len(counts) == 12
    assert counts.max() - counts.min() <= 1
    assert str(df[VINTAGE_COLUMN].min()) == "2020-01"
    assert len(df) == len(credit_df)


def test_random_assignment_is_independent_of_attributes(credit_df):
    df = add_vintage(credit_df, n_months=4, shift=0.0)
    means = df.groupby(VINTAGE_COLUMN)["age"].mean()
    assert means.max() - means.min() < 3.0


def test_shift_moves_driver_over_time(credit_df):
    df = add_vintage(credit_df, n_months=4, shift=2.0)
    means = df.groupby(VINTAGE_COLUMN)["rev_util"].mean()
    assert means.is_monotonic_increasing


def test_vintage_argument_checks(credit_df):
    with pytest.raises(ValueError):
        add_vintage(credit_df, n_months=1)
    with pytest.raises(ValueError):
        add_vintage(credit_df, shift=1.0, driver="missing")


def test_out_of_time_split(credit_df):
    df = add_vintage(credit_df, start="2019-01", n_months=12)
    dev, oot = out_of_time_split(df, "2019-10")
    assert len(dev) + len(oot) == len(df)
    assert dev[VINTAGE_COLUMN].max() < oot[VINTAGE_COLUMN].min()
    with pytest.raises(ValueError):
        out_of_time_split(df, "2030-01")
    with pytest.raises(ValueError):
        out_of_time_split(credit_df, "2019-10")


def test_oot_validation_detects_drift(credit_df):
    stable = oot_validation(add_vintage(credit_df, shift=0.0), "2020-07")
    drifted = oot_validation(add_vintage(credit_df, shift=2.0), "2020-07")
    assert list(stable.summary["sample"]) == ["in_time_holdout", "out_of_time"]
    assert stable.score_stability["status"] == "stable"
    assert drifted.score_stability["psi"] > stable.score_stability["psi"]
    periods = set(stable.by_vintage["period"])
    assert periods == {"in_time", "out_of_time"}
    ci = stable.summary[["auc_ci_low", "roc_auc", "auc_ci_high"]]
    assert (ci.diff(axis=1).iloc[:, 1:] >= 0).all().all()


def test_oot_cli(tmp_path):
    data = tmp_path / "s.csv"
    main(["synthesize", str(data), "--n", "900"])
    out = tmp_path / "oot"
    rc = main(["oot", str(data), "--cutoff", "2020-07", "--output-dir", str(out)])
    assert rc == 0
    summary = pd.read_csv(out / "oot_summary.csv")
    assert summary["roc_auc"].between(0.5, 1.0).all()
    assert (out / "oot_by_vintage.csv").exists()
