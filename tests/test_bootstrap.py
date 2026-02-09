import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk.cli import main
from credit_risk.config import CostConfig
from credit_risk.evaluation.bootstrap import (
    DEFAULT_METRICS,
    bootstrap_indices,
    decision_metrics,
    format_interval,
    metric_intervals,
)
from credit_risk.evaluation.compare import compare_models_with_intervals
from credit_risk.evaluation.delong import auc_confidence_interval
from credit_risk.features.engineering import build_xy


def _scores(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n)
    p = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 1.2 * (y - 0.3))))
    return y, p


def test_stratified_indices_keep_class_counts():
    y = np.array([0] * 70 + [1] * 30)
    idx = bootstrap_indices(y, 50, stratified=True, seed=1)
    assert idx.shape == (50, 100)
    assert (y[idx].sum(axis=1) == 30).all()


def test_intervals_contain_estimates():
    y, p = _scores()
    table = metric_intervals(y, p, n_boot=300)
    assert list(table["metric"]) == list(DEFAULT_METRICS)
    assert (table["ci_low"] <= table["estimate"]).all()
    assert (table["estimate"] <= table["ci_high"]).all()
    assert (table["se"] > 0).all()


def test_auc_interval_agrees_with_delong():
    y, p = _scores(3000)
    row = metric_intervals(y, p, {"roc_auc": DEFAULT_METRICS["roc_auc"]}, 1000)
    _, lo, hi = auc_confidence_interval(y, p)
    assert row["ci_low"].iloc[0] == pytest.approx(lo, abs=0.01)
    assert row["ci_high"].iloc[0] == pytest.approx(hi, abs=0.01)


def test_basic_interval_reflects_percentile():
    y, p = _scores()
    metrics = {"roc_auc": DEFAULT_METRICS["roc_auc"]}
    pct = metric_intervals(y, p, metrics, 200, method="percentile").iloc[0]
    basic = metric_intervals(y, p, metrics, 200, method="basic").iloc[0]
    est = pct["estimate"]
    assert basic["ci_low"] == pytest.approx(2 * est - pct["ci_high"])
    assert basic["ci_high"] == pytest.approx(2 * est - pct["ci_low"])


def test_invalid_arguments():
    y, p = _scores(200)
    with pytest.raises(ValueError):
        metric_intervals(y, p, method="bca")
    with pytest.raises(ValueError):
        metric_intervals(y, p, alpha=1.5)


def test_decision_metrics_match_direct_computation():
    y, p = _scores()
    cost = CostConfig()
    m = decision_metrics(0.4, cost)
    approved = p < 0.4
    assert m["approval_rate"](y, p) == pytest.approx(approved.mean())
    assert m["bad_rate_approved"](y, p) == pytest.approx(y[approved].mean())
    profit = np.where(y == 1, -cost.lgd, cost.margin)[approved].sum() / len(y)
    assert m["profit_per_applicant"](y, p) == pytest.approx(profit)


def test_format_interval():
    y, p = _scores(300)
    row = metric_intervals(y, p, n_boot=50).iloc[0]
    text = format_interval(row)
    assert text.startswith(f"{roc_auc_score(y, p):.3f} [")


def test_comparison_intervals(credit_df):
    X, y = build_xy(credit_df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    table, ci = compare_models_with_intervals(
        ["logistic", "hist_gbm"], X_tr, y_tr, X_te, y_te, n_boot=100
    )
    assert len(table) == 2
    assert set(ci["model"]) == {"logistic", "hist_gbm"}
    assert {"roc_auc", "brier", "profit_per_applicant"} <= set(ci["metric"])
    assert (ci["ci_low"] <= ci["ci_high"]).all()


def test_compare_cli_writes_intervals(tmp_path):
    data = tmp_path / "s.csv"
    main(["synthesize", str(data), "--n", "600"])
    args = ["compare", "--data", str(data), "--output-dir", str(tmp_path)]
    rc = main([*args, "--models", "logistic", "--n-boot", "50"])
    assert rc == 0
    assert (tmp_path / "model_comparison_ci.csv").exists()
