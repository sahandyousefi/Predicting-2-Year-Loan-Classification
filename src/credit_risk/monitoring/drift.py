from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def _edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    uniq = np.unique(reference[~np.isnan(reference)])
    if len(uniq) <= n_bins:
        # Discrete variable: one bin per observed value.
        return (uniq[:-1] + uniq[1:]) / 2
    return np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))


def psi(reference, current, n_bins: int = 10) -> float:
    """Population Stability Index.

    .. math:: \\mathrm{PSI} = \\sum_i (a_i - e_i) \\ln(a_i / e_i)

    where :math:`e_i` and :math:`a_i` are bin shares in the reference and
    current samples. Bins are quantiles of the reference sample.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = _edges(ref, n_bins)
    e = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1)
    a = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1)
    e = np.clip(e / e.sum(), EPS, None)
    a = np.clip(a / a.sum(), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float) -> str:
    """Conventional reading: < 0.1 stable, 0.1 to 0.25 moderate, above significant."""
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Characteristic stability (PSI per input) for each feature."""
    features = features or [c for c in reference.columns if c in current.columns]
    rows = [
        {
            "feature": f,
            "psi": round(psi(reference[f], current[f], n_bins), 4),
            "reference_mean": float(reference[f].mean()),
            "current_mean": float(current[f].mean()),
        }
        for f in features
    ]
    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    report["status"] = report["psi"].map(psi_status)
    return report.reset_index(drop=True)


def score_psi(reference_pd, current_pd, n_bins: int = 10) -> dict[str, float | str]:
    """Stability of the model output distribution."""
    value = psi(reference_pd, current_pd, n_bins)
    return {"psi": round(value, 4), "status": psi_status(value)}
