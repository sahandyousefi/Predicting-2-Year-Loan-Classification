"""Monotonic constraints for gradient boosting models.

A credit model is expected to move in a known direction for several inputs:
more past delinquency or higher utilization should never lower the PD, and
higher income should never raise it. Tree ensembles fitted without
constraints can break these relations in sparse regions of the data, which
makes reason codes contradictory and is hard to defend in model validation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# +1: PD non-decreasing in the feature, -1: non-increasing, 0: unconstrained.
EXPECTED_DIRECTION: dict[str, int] = {
    "age": -1,
    "debt_ratio": 1,
    "monthly_inc": -1,
    "open_credit": 0,
    "rev_util": 1,
    "dependents": 0,
    "debt_per_dependent": 1,
    "inc_per_dependent": -1,
    "credit_to_income": 0,
    "late_severity": 1,
    "log_monthly_inc": -1,
    "log_rev_util": 1,
    "log_debt_ratio": 1,
    "late_30_59": 1,
    "late_60_89": 1,
    "late_90": 1,
    "real_estate": 0,
}


def constraint_vector(
    columns: Sequence[str], overrides: dict[str, int] | None = None
) -> list[int]:
    """Direction for each column, in column order; unknown columns get 0."""
    directions = {**EXPECTED_DIRECTION, **(overrides or {})}
    bad = {k: v for k, v in directions.items() if v not in (-1, 0, 1)}
    if bad:
        raise ValueError(f"directions must be -1, 0 or 1: {bad}")
    return [directions.get(c, 0) for c in columns]


def model_kwargs(name: str, constraints: Sequence[int]) -> dict:
    """Translate a constraint vector into the keyword each library expects."""
    constraints = [int(c) for c in constraints]
    if name == "hist_gbm":
        return {"monotonic_cst": constraints}
    if name == "xgboost":
        return {"monotone_constraints": "(" + ",".join(map(str, constraints)) + ")"}
    if name == "lightgbm":
        return {"monotone_constraints": constraints}
    raise ValueError(f"monotonic constraints are not supported for '{name}'")


def monotonicity_violations(
    model,
    X: pd.DataFrame,
    feature: str,
    direction: int,
    n_grid: int = 20,
    n_rows: int = 200,
    tol: float = 1e-9,
    seed: int = 0,
) -> float:
    """Share of ICE curves that move against the expected direction.

    Each sampled row is evaluated on a quantile grid of ``feature`` with all
    other inputs held fixed. A curve violates the constraint if any step
    changes the PD by more than ``tol`` in the wrong direction.

    The check varies one column at a time, so it tests the fitted function,
    not the economic relation: engineered features derived from the same
    attribute are not moved together.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    rng = np.random.default_rng(seed)
    rows = X.iloc[rng.choice(len(X), size=min(n_rows, len(X)), replace=False)]
    grid = np.unique(np.quantile(X[feature], np.linspace(0.01, 0.99, n_grid)))
    repeated = rows.loc[rows.index.repeat(len(grid))].copy()
    repeated[feature] = np.tile(grid, len(rows))
    pd_grid = model.predict_proba(repeated)[:, 1].reshape(len(rows), len(grid))
    steps = np.diff(pd_grid, axis=1) * direction
    return float((steps < -tol).any(axis=1).mean())


def violation_table(
    model, X: pd.DataFrame, directions: dict[str, int] | None = None, **kwargs
) -> pd.DataFrame:
    """Violation share for every constrained column present in ``X``."""
    directions = directions or EXPECTED_DIRECTION
    rows = [
        {
            "feature": col,
            "direction": d,
            "violation_share": monotonicity_violations(model, X, col, d, **kwargs),
        }
        for col, d in directions.items()
        if col in X.columns and d != 0
    ]
    return pd.DataFrame(rows)
