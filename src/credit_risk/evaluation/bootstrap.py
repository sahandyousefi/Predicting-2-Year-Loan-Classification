"""Bootstrap confidence intervals for evaluation metrics.

All metrics are computed on the same bootstrap samples, so intervals of
different metrics, and of different models scored on the same applicants,
come from one set of resamples and can be compared directly. Resampling is
stratified by default: the number of defaulters is fixed in every replicate,
as it is in a stratified test split, and rank metrics are always defined.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credit_risk.config import CostConfig
from credit_risk.evaluation.calibration import expected_calibration_error
from credit_risk.evaluation.metrics import ks_statistic

Metric = Callable[[np.ndarray, np.ndarray], float]

DEFAULT_METRICS: dict[str, Metric] = {
    "roc_auc": lambda y, p: float(roc_auc_score(y, p)),
    "gini": lambda y, p: float(2 * roc_auc_score(y, p) - 1),
    "ks": ks_statistic,
    "pr_auc": lambda y, p: float(average_precision_score(y, p)),
    "brier": lambda y, p: float(brier_score_loss(y, p)),
    "ece": expected_calibration_error,
}


def decision_metrics(threshold: float, cost: CostConfig) -> dict[str, Metric]:
    """Approval rate, bad rate and profit per applicant at a fixed cut-off.

    The cut-off is held fixed across replicates, so the interval reflects
    sampling of applicants and not the uncertainty of choosing the cut-off.
    """

    def approval_rate(y: np.ndarray, p: np.ndarray) -> float:
        return float((p < threshold).mean())

    def bad_rate_approved(y: np.ndarray, p: np.ndarray) -> float:
        approved = p < threshold
        return float(y[approved].mean()) if approved.any() else 0.0

    def profit_per_applicant(y: np.ndarray, p: np.ndarray) -> float:
        gain = np.where(y == 1, -cost.lgd, cost.margin) * cost.ead
        return float(gain[p < threshold].sum() / len(y))

    return {
        "approval_rate": approval_rate,
        "bad_rate_approved": bad_rate_approved,
        "profit_per_applicant": profit_per_applicant,
    }


def bootstrap_indices(
    y: np.ndarray, n_boot: int, stratified: bool = True, seed: int = 0
) -> np.ndarray:
    """Array of shape ``(n_boot, n)`` with resampled row positions."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if not stratified:
        return rng.integers(0, len(y), size=(n_boot, len(y)))
    parts = []
    for cls in np.unique(y):
        pos = np.flatnonzero(y == cls)
        parts.append(pos[rng.integers(0, len(pos), size=(n_boot, len(pos)))])
    return np.concatenate(parts, axis=1)


def metric_intervals(
    y_true,
    y_prob,
    metrics: Mapping[str, Metric] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    method: str = "percentile",
    stratified: bool = True,
    seed: int = 0,
    indices: np.ndarray | None = None,
) -> pd.DataFrame:
    """Point estimate, bootstrap standard error and interval for each metric.

    Parameters
    ----------
    y_true, y_prob
        Labels and predicted PDs.
    metrics
        Mapping from name to ``f(y, p) -> float``. Defaults to discrimination
        and calibration metrics.
    n_boot
        Number of bootstrap replicates.
    alpha
        One minus the nominal coverage.
    method
        ``"percentile"`` uses the empirical quantiles of the replicates;
        ``"basic"`` reflects them around the estimate, ``2*theta - q``, which
        corrects a shift between the bootstrap and sampling distributions.
    stratified
        Resample within each class.
    seed
        Seed of the resampling.
    indices
        Precomputed indices from :func:`bootstrap_indices`; pass the same array
        for several models to obtain paired intervals.

    Returns
    -------
    pandas.DataFrame
        One row per metric with ``estimate``, ``se``, ``ci_low`` and
        ``ci_high``.
    """
    if method not in {"percentile", "basic"}:
        raise ValueError("method must be 'percentile' or 'basic'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    metrics = metrics or DEFAULT_METRICS
    if indices is None:
        indices = bootstrap_indices(y, n_boot, stratified, seed)

    rows = []
    for name, fn in metrics.items():
        estimate = fn(y, p)
        reps = np.array([fn(y[idx], p[idx]) for idx in indices])
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
        if method == "basic":
            lo, hi = 2 * estimate - hi, 2 * estimate - lo
        rows.append(
            {
                "metric": name,
                "estimate": estimate,
                "se": float(reps.std(ddof=1)),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )
    return pd.DataFrame(rows)


def format_interval(row: pd.Series, digits: int = 3) -> str:
    return (
        f"{row['estimate']:.{digits}f} "
        f"[{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"
    )
