from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, rankdata


@dataclass(frozen=True)
class DeLongResult:
    auc_a: float
    auc_b: float
    diff: float
    se: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float


def _structural_components(
    pos: np.ndarray, neg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Placement values for k score vectors using midranks (Sun and Xu, 2014).

    ``pos`` and ``neg`` have shape (k, m) and (k, n).
    """
    k, m = pos.shape
    n = neg.shape[1]
    tx = np.vstack([rankdata(row) for row in pos])
    ty = np.vstack([rankdata(row) for row in neg])
    tz = np.vstack([rankdata(row) for row in np.hstack([pos, neg])])
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return auc, v01, v10


def delong_test(y_true, score_a, score_b, alpha: float = 0.05) -> DeLongResult:
    """DeLong test for two correlated ROC curves on the same sample.

    Returns both AUCs, their difference, its standard error, the two-sided
    p-value and a Wald confidence interval for the difference.
    """
    y = np.asarray(y_true).astype(bool)
    scores = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        raise ValueError("need at least two positives and two negatives")

    auc, v01, v10 = _structural_components(pos, neg)
    cov = np.cov(v01) / m + np.cov(v10) / n
    contrast = np.array([1.0, -1.0])
    var = float(contrast @ cov @ contrast)
    se = np.sqrt(max(var, 1e-300))
    diff = float(auc[0] - auc[1])
    z = diff / se
    p = float(2 * norm.sf(abs(z)))
    half = norm.ppf(1 - alpha / 2) * se
    return DeLongResult(
        auc_a=float(auc[0]),
        auc_b=float(auc[1]),
        diff=diff,
        se=float(se),
        z=float(z),
        p_value=p,
        ci_low=diff - half,
        ci_high=diff + half,
    )


def auc_confidence_interval(
    y_true, y_score, alpha: float = 0.05
) -> tuple[float, float, float]:
    """AUC with a DeLong standard-error confidence interval."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(y_score, float)[None, :]
    auc, v01, v10 = _structural_components(s[:, y], s[:, ~y])
    var = v01.var(ddof=1) / y.sum() + v10.var(ddof=1) / (~y).sum()
    half = norm.ppf(1 - alpha / 2) * np.sqrt(var)
    a = float(auc[0])
    return a, max(0.0, a - half), min(1.0, a + half)
