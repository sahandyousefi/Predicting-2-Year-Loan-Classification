from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def feature_grid(x: pd.Series, n_points: int = 20, trim: float = 0.01) -> np.ndarray:
    """Quantile grid that ignores the extreme tails."""
    qs = np.linspace(trim, 1 - trim, n_points)
    return np.unique(np.quantile(x.dropna(), qs))


def ice_curves(
    model,
    X_raw: pd.DataFrame,
    feature: str,
    grid: np.ndarray | None = None,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    sample: int | None = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Individual conditional expectation of PD for one raw feature.

    ``transform`` rebuilds model inputs from raw columns, so engineered
    features derived from ``feature`` move together with it. Rows are
    applicants and columns are grid values.
    """
    if sample is not None and len(X_raw) > sample:
        X_raw = X_raw.sample(sample, random_state=seed)
    grid = feature_grid(X_raw[feature]) if grid is None else np.asarray(grid)
    transform = transform or (lambda df: df)

    curves = np.empty((len(X_raw), len(grid)))
    for j, value in enumerate(grid):
        X_mod = X_raw.copy()
        X_mod[feature] = value
        curves[:, j] = model.predict_proba(transform(X_mod))[:, 1]
    return pd.DataFrame(curves, index=X_raw.index, columns=grid)


def partial_dependence(ice: pd.DataFrame) -> pd.Series:
    return ice.mean(axis=0)


def centered_ice(ice: pd.DataFrame) -> pd.DataFrame:
    """ICE curves anchored at the first grid value, which makes
    heterogeneous effects easier to see."""
    return ice.sub(ice.iloc[:, 0], axis=0)


def interaction_strength(ice: pd.DataFrame) -> float:
    """Spread of centered ICE curves around their mean at the last grid point.

    Zero means the feature acts additively on PD across applicants.
    """
    c = centered_ice(ice)
    return float(c.iloc[:, -1].std())
