"""Synthetic free-text loan-purpose descriptions.

The benchmark dataset has no text field. To develop and test the text
pipeline, this module attaches a short purpose statement to each row. The
purpose category is drawn with a propensity that depends on the outcome, so
the text carries signal beyond the structured attributes. The strength of that
dependence is a parameter; any result obtained on this text describes the
generator, not real applicants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.data.schema import TARGET

TEXT_COLUMN = "purpose_text"
_TEXT_STREAM = 101

# (prior share, log-odds shift towards default)
PURPOSES: dict[str, tuple[float, float]] = {
    "home_improvement": (0.14, -0.7),
    "car": (0.14, -0.4),
    "education": (0.08, -0.2),
    "major_purchase": (0.08, -0.2),
    "wedding": (0.05, 0.0),
    "moving": (0.05, 0.1),
    "debt_consolidation": (0.24, 0.3),
    "small_business": (0.06, 0.4),
    "medical": (0.07, 0.5),
    "overdue_bills": (0.05, 0.9),
    "emergency": (0.04, 0.8),
}

TEMPLATES: dict[str, list[str]] = {
    "home_improvement": [
        "renovating the kitchen in the house we own",
        "new roof and windows for our home",
        "finishing the basement before we sell",
    ],
    "car": [
        "buying a used car, part of the price already saved",
        "replacing my car which is over ten years old",
        "car purchase to commute to a new job",
    ],
    "education": [
        "tuition for a professional certification course",
        "paying for my last semester of college",
        "evening classes to finish a degree",
    ],
    "major_purchase": [
        "new furniture and appliances for the apartment",
        "buying a piano for my daughter",
        "computer equipment for work from home",
    ],
    "wedding": [
        "covering part of our wedding costs",
        "wedding venue and catering deposit",
    ],
    "moving": [
        "moving costs for a job in another state",
        "deposit and first month rent on a new place",
    ],
    "debt_consolidation": [
        "consolidate three credit cards into one payment",
        "pay off high interest cards with a lower rate",
        "combine my store cards and personal loan",
    ],
    "small_business": [
        "inventory for my small online shop",
        "equipment to start a food truck",
        "working capital while the business grows",
    ],
    "medical": [
        "hospital bills not covered by insurance",
        "dental surgery and follow up treatment",
        "medical expenses after an accident",
    ],
    "overdue_bills": [
        "catch up on rent and utilities that are past due",
        "pay collections and late bills",
        "behind on payments and need to get current",
    ],
    "emergency": [
        "urgent cash after losing hours at work",
        "emergency repairs and bills this month",
        "need money fast to cover expenses until next paycheck",
    ],
}

QUALIFIERS_LOW = ["", "", "", "stable job for six years", "have savings as backup"]
QUALIFIERS_HIGH = ["", "", "", "income dropped recently", "had some missed payments"]


def make_purpose_text(
    y: pd.Series | np.ndarray, signal: float = 1.0, seed: int = 0
) -> pd.Series:
    """Draw one purpose statement per outcome.

    Parameters
    ----------
    y : array-like of {0, 1}
        Outcomes the statements are conditioned on.
    signal : float
        Multiplier of the log-odds shifts. Zero makes the text independent of
        the outcome, which is the null case of an ablation.
    seed : int
        Random seed.
    """
    # A separate stream keeps the text independent of the attribute draws made
    # by make_synthetic with the same seed.
    rng = np.random.default_rng([seed, _TEXT_STREAM])
    y = np.asarray(y, dtype=float)
    names = list(PURPOSES)
    prior = np.array([PURPOSES[k][0] for k in names])
    shift = np.array([PURPOSES[k][1] for k in names])
    centred = 2.0 * y[:, None] - 1.0
    logits = np.log(prior)[None, :] + signal * shift[None, :] * centred
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    cum = probs.cumsum(axis=1)
    draws = (rng.uniform(size=(len(y), 1)) > cum).sum(axis=1)

    texts = []
    for yi, k in zip(y, draws):
        base = rng.choice(TEMPLATES[names[k]])
        use_high = rng.uniform() < 0.5 + 0.25 * signal * (2 * yi - 1)
        qual = rng.choice(QUALIFIERS_HIGH if use_high else QUALIFIERS_LOW)
        texts.append(f"{base}, {qual}" if qual else base)
    return pd.Series(texts, name=TEXT_COLUMN)


def add_purpose_text(
    df: pd.DataFrame, signal: float = 1.0, seed: int = 0
) -> pd.DataFrame:
    """Return a copy of a labeled frame with a ``purpose_text`` column."""
    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' is required to condition the text")
    out = df.copy()
    out[TEXT_COLUMN] = make_purpose_text(df[TARGET], signal, seed).to_numpy()
    return out
