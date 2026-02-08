import numpy as np
import pandas as pd
import pytest

from credit_risk.cli import main
from credit_risk.data.synthetic_text import add_purpose_text
from credit_risk.evaluation.ablation import (
    build_text_pipeline,
    corrected_resampled_ttest,
    signal_sweep,
    text_ablation,
)


def test_corrected_ttest_is_more_conservative():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.01, 10)
    stat, p = corrected_resampled_ttest(d, n_train=800, n_test=200)
    naive = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert abs(stat) < abs(naive)
    assert 0.0 <= p <= 1.0
    assert corrected_resampled_ttest(np.zeros(5), 80, 20) == (0.0, 1.0)


def test_pipeline_validates_arguments():
    with pytest.raises(ValueError):
        build_text_pipeline("all")
    with pytest.raises(ValueError):
        build_text_pipeline("text", learner="svm")


def test_ablation_requires_text(credit_df):
    with pytest.raises(ValueError):
        text_ablation(credit_df)


def test_text_ablation_table(credit_df):
    df = add_purpose_text(credit_df, signal=1.0)
    table = text_ablation(df, n_components=(4,), n_splits=3, n_repeats=1)
    assert list(table["feature_set"]) == ["structured", "text", "combined"]
    base = table.set_index("feature_set").loc["structured"]
    assert base["gain_vs_structured"] == 0.0 and np.isnan(base["p_value"])
    assert table["auc_mean"].between(0.5, 1.0).all()


def test_signal_sweep_negative_control(credit_df):
    table = signal_sweep(
        credit_df, signals=(0.0, 1.0), n_components=(8,), n_splits=3, n_repeats=1
    )
    combined = table[table["feature_set"] == "combined"].set_index("signal")
    assert combined.loc[1.0, "gain_vs_structured"] > 0.02
    assert abs(combined.loc[0.0, "gain_vs_structured"]) < 0.02


def test_text_ablation_cli(tmp_path):
    out = tmp_path / "ablation.csv"
    args = ["text-ablation", "--n", "600", "--signals", "0", "--learners"]
    args += ["logistic", "--dims", "4", "--folds", "3", "--repeats", "1"]
    assert main([*args, "--output", str(out)]) == 0
    assert len(pd.read_csv(out)) == 3
