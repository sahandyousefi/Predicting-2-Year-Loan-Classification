"""Synthetic application vintages.

The benchmark has no application date. This module assigns each row an
application month so that out-of-time validation can be developed and tested.
With ``shift = 0`` months are assigned at random and an out-of-time split is
equivalent to a random split. With ``shift > 0`` later months receive
borrowers with higher values of ``driver``, which imitates gradual population
drift such as rising revolving utilization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VINTAGE_COLUMN = "vintage"
_VINTAGE_STREAM = 202


def add_vintage(
    df: pd.DataFrame,
    start: str = "2019-01",
    n_months: int = 24,
    shift: float = 0.0,
    driver: str = "rev_util",
    seed: int = 0,
) -> pd.DataFrame:
    """Return a copy with a monthly ``vintage`` column (``pandas.Period``).

    Parameters
    ----------
    df : DataFrame
        Applicant data.
    start : str
        First month, ``YYYY-MM``.
    n_months : int
        Number of monthly vintages; rows are spread evenly across them.
    shift : float
        Correlation strength between the vintage order and the standardized
        rank of ``driver``. Zero gives random assignment.
    driver : str
        Column whose distribution moves over time when ``shift > 0``.
    seed : int
        Random seed.
    """
    if n_months < 2:
        raise ValueError("n_months must be at least 2")
    if shift and driver not in df.columns:
        raise ValueError(f"driver column '{driver}' not found")
    # A separate stream keeps the assignment independent of the attribute
    # draws made by make_synthetic with the same seed.
    rng = np.random.default_rng([seed, _VINTAGE_STREAM])
    n = len(df)
    noise = rng.normal(size=n)
    if shift:
        ranks = df[driver].rank(method="first").to_numpy()
        z = (ranks - ranks.mean()) / ranks.std()
        key = shift * z + noise
    else:
        key = noise
    order = np.argsort(np.argsort(key, kind="stable"), kind="stable")
    month = order * n_months // n
    out = df.copy()
    out[VINTAGE_COLUMN] = pd.period_range(start, periods=n_months, freq="M")[month]
    return out


def out_of_time_split(
    df: pd.DataFrame, cutoff: str, column: str = VINTAGE_COLUMN
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development sample before ``cutoff`` and out-of-time sample from it."""
    if column not in df.columns:
        raise ValueError(f"'{column}' column is required")
    period = pd.Period(cutoff, freq="M")
    months = df[column].astype("period[M]")
    dev, oot = df[months < period], df[months >= period]
    if dev.empty or oot.empty:
        raise ValueError(f"cutoff {cutoff} leaves an empty sample")
    return dev, oot
