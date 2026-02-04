from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def reliability_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability against observed rate in quantile bins."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    table = df.groupby("bin").agg(
        count=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")
    )
    return table.reset_index()


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    t = reliability_table(y_true, y_prob, n_bins)
    weights = t["count"] / t["count"].sum()
    return float((weights * (t["mean_pred"] - t["observed"]).abs()).sum())


def hosmer_lemeshow(y_true, y_prob, n_bins: int = 10) -> tuple[float, float]:
    """Hosmer-Lemeshow statistic and p-value with ``n_bins - 2`` dof.

    A small p-value is evidence that predicted probabilities are miscalibrated.
    """
    t = reliability_table(y_true, y_prob, n_bins)
    expected = t["count"] * t["mean_pred"]
    observed = t["count"] * t["observed"]
    variance = t["count"] * t["mean_pred"] * (1 - t["mean_pred"])
    stat = float(((observed - expected) ** 2 / np.maximum(variance, 1e-12)).sum())
    return stat, float(chi2.sf(stat, n_bins - 2))


def calibration_summary(y_true, y_prob, n_bins: int = 10) -> dict[str, float]:
    hl_stat, hl_p = hosmer_lemeshow(y_true, y_prob, n_bins)
    return {
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "ece": round(expected_calibration_error(y_true, y_prob, n_bins), 4),
        "hl_stat": round(hl_stat, 3),
        "hl_pvalue": round(hl_p, 4),
    }


def calibrate(estimator, X_cal, y_cal, method: str = "isotonic"):
    """Fit a calibration map on held-out data for an already fitted model.

    ``method`` is ``"sigmoid"`` (Platt scaling) or ``"isotonic"``.
    """
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'")
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except ImportError:  # scikit-learn < 1.6
        calibrated = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    return calibrated.fit(X_cal, y_cal)
