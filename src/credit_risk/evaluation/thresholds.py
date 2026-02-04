from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_risk.config import CostConfig


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    approval_rate: float
    bad_rate_approved: float
    profit_per_applicant: float


def breakeven_pd(cost: CostConfig) -> float:
    """PD at which the expected value of approving one loan is zero.

    Approving earns ``margin * EAD`` with probability ``1 - PD`` and loses
    ``LGD * EAD`` with probability ``PD``, so the break-even point is
    ``margin / (margin + LGD)``. With calibrated PDs this is the
    decision-theoretic cut-off.
    """
    return cost.margin / (cost.margin + cost.lgd)


def profit_curve(y_true, pd_hat, cost: CostConfig, n_points: int = 201) -> pd.DataFrame:
    """Realized profit per applicant when approving everyone with PD below t."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pd_hat, dtype=float)
    gain = np.where(y == 1, -cost.lgd, cost.margin) * cost.ead
    rows = []
    for t in np.linspace(0, 1, n_points):
        approved = p < t
        n_app = approved.sum()
        rows.append(
            {
                "threshold": t,
                "approval_rate": approved.mean(),
                "bad_rate_approved": y[approved].mean() if n_app else 0.0,
                "profit_per_applicant": gain[approved].sum() / len(y),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(y_true, pd_hat, cost: CostConfig) -> ThresholdResult:
    curve = profit_curve(y_true, pd_hat, cost)
    best = curve.loc[curve["profit_per_applicant"].idxmax()]
    return ThresholdResult(
        threshold=float(best["threshold"]),
        approval_rate=float(best["approval_rate"]),
        bad_rate_approved=float(best["bad_rate_approved"]),
        profit_per_applicant=float(best["profit_per_applicant"]),
    )
