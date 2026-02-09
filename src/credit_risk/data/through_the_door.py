"""Synthetic through-the-door population for reject inference studies.

A lender only observes outcomes for applicants it approved. To study how
much that selection biases a model, and whether reject inference helps, the
generator produces every applicant with an outcome and then applies a
historical approval policy. The outcomes of rejected applicants are kept in
the frame as ground truth for evaluation and must not be used in fitting.

The historical policy scores applicants on a subset of the structured inputs
plus a hidden variable that is correlated with default and is not in the
data, standing in for information an underwriter saw (for example a bank
statement). With ``hidden_signal = 0`` selection depends on observed inputs
only (missing at random); with ``hidden_signal > 0`` it also depends on the
outcome through the hidden variable (missing not at random).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_risk.data.schema import TARGET
from credit_risk.data.synthetic import SyntheticSpec, make_synthetic

ACCEPTED = "accepted"


@dataclass(frozen=True)
class ThroughTheDoorSpec:
    n: int = 20000
    base_rate: float = 0.12
    approval_rate: float = 0.6
    hidden_signal: float = 0.25
    policy_noise: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.approval_rate < 1:
            raise ValueError("approval_rate must be in (0, 1)")
        if self.hidden_signal < 0 or self.policy_noise < 0:
            raise ValueError("hidden_signal and policy_noise must be non-negative")


def _standardize(x: pd.Series) -> np.ndarray:
    x = x.to_numpy(dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def make_through_the_door(spec: ThroughTheDoorSpec | None = None) -> pd.DataFrame:
    """All applicants with outcomes and the historical decision in ``accepted``."""
    spec = spec or ThroughTheDoorSpec()
    df = make_synthetic(
        SyntheticSpec(n=spec.n, base_rate=spec.base_rate, seed=spec.seed)
    )
    rng = np.random.default_rng(spec.seed + 1)
    y = df[TARGET].to_numpy()

    late = df["late_30_59"] + 2 * df["late_60_89"] + 3 * df["late_90"]
    hidden = spec.hidden_signal * (2 * y - 1) + rng.normal(0, 1, len(df))
    policy_score = (
        _standardize(np.log1p(late))
        + 0.8 * _standardize(df["rev_util"])
        - 0.5 * _standardize(np.log(df["monthly_inc"]))
        + 0.7 * hidden
        + rng.normal(0, spec.policy_noise, len(df))
    )
    cutoff = np.quantile(policy_score, spec.approval_rate)
    df[ACCEPTED] = policy_score <= cutoff
    return df
