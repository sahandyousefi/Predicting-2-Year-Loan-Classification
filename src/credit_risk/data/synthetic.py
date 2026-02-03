from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from credit_risk.data.schema import FEATURES, TARGET


@dataclass(frozen=True)
class SyntheticSpec:
    """Parameters for a synthetic sample with the benchmark schema.

    The generator is used for tests and CI only. Its coefficients follow the
    directions seen in the real data (younger borrowers, high utilization and
    past delinquency raise risk) but the magnitudes are not estimates.
    """

    n: int = 2000
    base_rate: float = 0.5
    seed: int = 0


def _late_counts(rng: np.random.Generator, n: int) -> pd.DataFrame:
    # A shared latent "payment stress" factor makes the three counts correlated.
    stress = rng.gamma(shape=0.3, scale=2.0, size=n)
    return pd.DataFrame(
        {
            "late_30_59": rng.poisson(0.6 * stress),
            "late_60_89": rng.poisson(0.25 * stress),
            "late_90": rng.poisson(0.3 * stress),
        }
    ).astype(float)


def make_synthetic(spec: SyntheticSpec | None = None) -> pd.DataFrame:
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n = spec.n

    age = np.clip(rng.normal(50, 14, n), 21, 95).round()
    monthly_inc = np.round(rng.lognormal(8.5, 0.55, n), 0)
    dependents = rng.choice(
        [0, 1, 2, 3, 4, 5], size=n, p=[0.5, 0.2, 0.15, 0.1, 0.04, 0.01]
    )
    rev_util = np.clip(rng.beta(0.7, 1.4, n) * 1.1, 0, None)
    debt_ratio = rng.lognormal(-1.2, 0.8, n)
    open_credit = rng.poisson(8, n).astype(float)
    real_estate = rng.poisson(1.0, n).astype(float)
    late = _late_counts(rng, n)

    score = (
        -0.035 * (age - 50)
        + 1.6 * (rev_util - 0.4)
        + 0.25 * np.log1p(debt_ratio)
        - 0.35 * (np.log(monthly_inc) - 8.5)
        + 0.1 * dependents
        + 0.45 * late["late_30_59"]
        + 0.6 * late["late_60_89"]
        + 0.8 * late["late_90"]
    ).to_numpy()
    # Solve for the intercept that hits the requested default rate.
    intercept = brentq(lambda b: expit(score + b).mean() - spec.base_rate, -20, 20)
    y = (rng.uniform(size=n) < expit(score + intercept)).astype(int)

    df = pd.DataFrame(
        {
            "rev_util": rev_util,
            "age": age,
            "late_30_59": late["late_30_59"],
            "debt_ratio": debt_ratio,
            "monthly_inc": monthly_inc,
            "open_credit": open_credit,
            "late_90": late["late_90"],
            "real_estate": real_estate,
            "late_60_89": late["late_60_89"],
            "dependents": dependents.astype(float),
        }
    )
    df[TARGET] = y
    return df[FEATURES + [TARGET]]
