import pytest

from credit_risk.features.engineering import build_xy

optuna = pytest.importorskip("optuna")

from credit_risk.models.tuning import nested_cv, tune  # noqa: E402


def test_tune_returns_valid_params(credit_df):
    X, y = build_xy(credit_df)
    params, score = tune("logistic", X, y, n_trials=4, cv_folds=3)
    assert params["C"] > 0
    assert 0.5 < score <= 1.0


def test_nested_cv_shapes(credit_df):
    X, y = build_xy(credit_df.iloc[:600])
    res = nested_cv("hist_gbm", X, y, outer_folds=3, inner_folds=2, n_trials=2)
    assert len(res.outer_scores) == 3 == len(res.params)
    assert 0.5 < res.mean < 1.0
