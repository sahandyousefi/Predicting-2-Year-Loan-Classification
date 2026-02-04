import numpy as np
from sklearn.model_selection import train_test_split

from credit_risk.data.schema import TARGET
from credit_risk.evaluation.calibration import (
    calibrate,
    calibration_summary,
    expected_calibration_error,
    hosmer_lemeshow,
    reliability_table,
)
from credit_risk.features.engineering import build_xy
from credit_risk.models.train import build_pipeline


def _calibrated_sample(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    return (rng.uniform(size=n) < p).astype(int), p


def test_well_calibrated_scores_pass():
    y, p = _calibrated_sample()
    assert expected_calibration_error(y, p) < 0.03
    assert hosmer_lemeshow(y, p)[1] > 0.01


def test_distorted_scores_fail():
    y, p = _calibrated_sample()
    distorted = p**3
    assert expected_calibration_error(y, distorted) > 0.1
    assert hosmer_lemeshow(y, distorted)[1] < 1e-6


def test_reliability_bins_cover_sample():
    y, p = _calibrated_sample(1000)
    assert reliability_table(y, p, 5)["count"].sum() == 1000


def test_calibration_reduces_ece_of_balanced_model(credit_df):
    # class_weight="balanced" inflates probabilities on an imbalanced sample.
    df = credit_df.copy()
    keep = (df[TARGET] == 0) | (np.arange(len(df)) % 4 == 0)
    X, y = build_xy(df[keep])
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=0
    )
    model = build_pipeline(C=1.0).fit(X_tr, y_tr)
    raw = model.predict_proba(X_cal)[:, 1]
    cal = calibrate(model, X_cal, y_cal, "sigmoid").predict_proba(X_cal)[:, 1]
    assert expected_calibration_error(y_cal, cal) < expected_calibration_error(
        y_cal, raw
    )
    assert set(calibration_summary(y_cal, cal)) == {
        "brier",
        "ece",
        "hl_stat",
        "hl_pvalue",
    }
