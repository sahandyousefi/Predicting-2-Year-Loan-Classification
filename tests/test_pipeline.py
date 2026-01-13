import json

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, add_engineered_features, build_xy
from src.train import best_f1_threshold, build_pipeline, evaluate, run


def make_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "rev_util": rng.uniform(0, 1.2, n),
            "age": rng.integers(21, 80, n).astype(float),
            "late_30_59": rng.poisson(0.3, n).astype(float),
            "debt_ratio": rng.exponential(0.4, n),
            "monthly_inc": rng.lognormal(8.5, 0.6, n),
            "open_credit": rng.integers(0, 25, n).astype(float),
            "late_90": rng.poisson(0.2, n).astype(float),
            "real_estate": rng.integers(0, 4, n).astype(float),
            "late_60_89": rng.poisson(0.1, n).astype(float),
            "dependents": rng.integers(0, 5, n).astype(float),
        }
    )
    logit = -0.04 * (df["age"] - 50) + 0.8 * df["late_90"] + 0.5 * df["late_30_59"]
    df["dlq_2yrs"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def test_engineered_columns_present():
    out = add_engineered_features(make_frame(50))
    assert set(FEATURE_COLUMNS).issubset(out.columns)


def test_late_severity_weights():
    df = make_frame(5)
    df[["late_30_59", "late_60_89", "late_90"]] = [[1, 1, 1]] * 5
    assert (add_engineered_features(df)["late_severity"] == 6).all()


def test_no_infinite_values():
    df = make_frame(50)
    df.loc[0, ["monthly_inc", "dependents"]] = [0, 0]
    X, _ = build_xy(df)
    assert np.isfinite(X.to_numpy()).all()


def test_missing_column_raises():
    with pytest.raises(ValueError):
        add_engineered_features(make_frame(10).drop(columns=["late_90"]))


def test_input_not_mutated():
    df = make_frame(20)
    before = df.copy()
    add_engineered_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_pipeline_and_threshold():
    X, y = build_xy(make_frame())
    pipe = build_pipeline().fit(X, y)
    prob = pipe.predict_proba(X)[:, 1]
    t = best_f1_threshold(y, prob)
    assert 0 < t < 1
    assert evaluate(y, prob, t)["roc_auc"] > 0.6


def test_run_writes_metrics(tmp_path):
    csv = tmp_path / "data.csv"
    make_frame(600).to_csv(csv, index=False)
    report = run(str(csv), output_dir=str(tmp_path / "artifacts"))
    saved = json.loads((tmp_path / "artifacts" / "metrics.json").read_text())
    assert saved["cv_roc_auc_mean"] == report["cv_roc_auc_mean"]
    assert len(saved["coefficients"]) == len(FEATURE_COLUMNS)
