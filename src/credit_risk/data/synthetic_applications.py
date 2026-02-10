"""Synthetic free-text loan applications with known field values.

Each application is a short first-person statement built from the benchmark
attributes of one synthetic borrower, a loan purpose and a few distracting
facts (job tenure, rent, requested amount). Every field is stated with one of
several phrasings, some of which the rule-based baseline is not written for,
and each field is left out with probability ``missing_rate``. The gold record
holds the stated value of each field, or ``None`` when it was left out. These
texts measure extraction on a known distribution; they are not a sample of
real applications.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_risk.data.synthetic import SyntheticSpec, make_synthetic
from credit_risk.data.synthetic_text import PURPOSES, TEMPLATES

_APPLICATION_STREAM = 404
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"]

Phrase = Callable[[np.random.Generator, float], str]


def _word(n: float) -> str:
    n = int(n)
    return WORDS[n] if n < len(WORDS) else str(n)


def _count(n: float, singular: str, plural: str) -> str:
    return f"{_word(n)} {singular if int(n) == 1 else plural}"


def _money(rng: np.random.Generator, x: float) -> str:
    style = rng.integers(3)
    if style == 2 and x >= 1000 and x % 100 == 0:
        return f"{x / 1000:.1f}k"
    if style == 1:
        return f"{x:.0f} dollars"
    return f"${x:,.0f}"


def _times(rng: np.random.Generator, n: float) -> str:
    n = int(n)
    if n == 0:
        return "zero times"
    if n == 1:
        return rng.choice(["once", "one time"])
    if n == 2:
        return rng.choice(["twice", "two times"])
    return f"{_word(n)} times"


PHRASES: dict[str, list[Phrase]] = {
    "age": [
        lambda r, v: f"I'm {v:.0f}",
        lambda r, v: f"I am {v:.0f} years old",
        lambda r, v: f"my age is {v:.0f}",
    ],
    "monthly_inc": [
        lambda r, v: f"my take-home pay is about {_money(r, v)} a month",
        lambda r, v: f"I earn {_money(r, v)} per month after tax",
        lambda r, v: f"I bring home around {_money(r, v)} every month",
    ],
    "monthly_debt": [
        lambda r, v: f"I pay {_money(r, v)} a month toward existing loans and cards",
        lambda r, v: f"my debt payments come to {_money(r, v)} per month",
        lambda r, v: f"about {_money(r, v)} of each month goes to what I owe",
    ],
    "dependents": [
        lambda r, v: f"I have {_count(v, 'dependent', 'dependents')}",
        lambda r, v: f"I support {_count(v, 'child', 'children')}",
        lambda r, v: (
            "one person relies on my income"
            if v == 1
            else f"{_word(v)} people rely on my income"
        ),
    ],
    "open_credit": [
        lambda r, v: f"I have {v:.0f} open credit accounts",
        lambda r, v: f"there are {v:.0f} active credit lines in my name",
        lambda r, v: f"my credit report shows {v:.0f} accounts I still use",
    ],
    "real_estate": [
        lambda r, v: f"I have {_count(v, 'mortgage', 'mortgages')}",
        lambda r, v: f"I hold {_count(v, 'real estate loan', 'real estate loans')}",
        lambda r, v: f"I am paying off {_count(v, 'home loan', 'home loans')}",
    ],
    "rev_util": [
        lambda r, v: f"my cards are at about {v * 100:.0f}% of their limits",
        lambda r, v: f"I am using {v * 100:.0f} percent of my revolving credit",
        lambda r, v: f"my balances sit near {v * 100:.0f}% of what I can borrow",
    ],
    "late_30_59": [
        lambda r, v: f"I was 30-59 days late {_times(r, v)}",
        lambda r, v: f"I paid 30 to 59 days past due {_times(r, v)}",
        lambda r, v: f"a bill slipped by a month or so {_times(r, v)}",
    ],
    "late_60_89": [
        lambda r, v: f"I was 60-89 days late {_times(r, v)}",
        lambda r, v: f"I fell 60 to 89 days past due {_times(r, v)}",
        lambda r, v: f"I was two months behind {_times(r, v)}",
    ],
    "late_90": [
        lambda r, v: f"I was 90 or more days late {_times(r, v)}",
        lambda r, v: f"I went 90+ days past due {_times(r, v)}",
        lambda r, v: f"an account was three months behind {_times(r, v)}",
    ],
}

DISTRACTORS: list[Callable[[np.random.Generator], str]] = [
    lambda r: f"I have been with my employer for {r.integers(2, 25)} years",
    lambda r: f"rent on my apartment is ${r.integers(6, 25) * 100:,} a month",
    lambda r: f"I am asking to borrow ${r.integers(2, 40) * 500:,}",
    lambda r: f"my credit score was around {r.integers(560, 800)} last time I checked",
]


@dataclass
class ApplicationSample:
    texts: list[str]
    gold: list[dict[str, float | str | None]]
    attributes: pd.DataFrame


def _sentence(clause: str) -> str:
    return clause[0].upper() + clause[1:] + "."


def make_applications(
    n: int = 200,
    missing_rate: float = 0.15,
    hard_rate: float = 0.3,
    seed: int = 0,
) -> ApplicationSample:
    """Generate applications and their gold field values.

    Parameters
    ----------
    n : int
        Number of applications.
    missing_rate : float
        Probability that a field is not mentioned.
    hard_rate : float
        Probability that a mentioned field uses the last phrasing in its list,
        which avoids the keywords the rule-based baseline relies on.
    seed : int
        Random seed.
    """
    attrs = make_synthetic(SyntheticSpec(n=n, base_rate=0.2, seed=seed))
    rng = np.random.default_rng([seed, _APPLICATION_STREAM])
    names = list(PURPOSES)
    prior = np.array([PURPOSES[k][0] for k in names])
    texts, gold = [], []
    for _, row in attrs.iterrows():
        record: dict[str, float | str | None] = {}
        values = {
            "age": float(row["age"]),
            "monthly_inc": float(row["monthly_inc"]),
            "monthly_debt": float(round(row["debt_ratio"] * row["monthly_inc"], -1)),
            "dependents": float(row["dependents"]),
            "open_credit": float(row["open_credit"]),
            "real_estate": float(row["real_estate"]),
            "rev_util": round(float(row["rev_util"]) * 100) / 100,
            "late_30_59": float(row["late_30_59"]),
            "late_60_89": float(row["late_60_89"]),
            "late_90": float(row["late_90"]),
        }
        clauses = []
        late = ("late_30_59", "late_60_89", "late_90")
        never_late = all(values[k] == 0 for k in late) and rng.uniform() < 0.7
        for field, value in values.items():
            if field in late and never_late:
                continue
            if rng.uniform() < missing_rate:
                record[field] = None
                continue
            options = PHRASES[field]
            k = (
                len(options) - 1
                if rng.uniform() < hard_rate
                else rng.integers(len(options) - 1)
            )
            if field == "monthly_inc":
                # Income is rounded as applicants state it; gold follows the text.
                value = float(round(value, -2))
            text = options[k](rng, value)
            record[field] = value
            clauses.append(text)
        if never_late:
            for k in late:
                record[k] = 0.0
            clauses.append("I have never been late on a payment")
        purpose = names[rng.choice(len(names), p=prior / prior.sum())]
        if rng.uniform() < missing_rate:
            record["purpose"] = None
        else:
            record["purpose"] = purpose
            clauses.append(f"purpose of the loan: {rng.choice(TEMPLATES[purpose])}")
        clauses += [DISTRACTORS[i](rng) for i in rng.permutation(len(DISTRACTORS))[:2]]
        order = rng.permutation(len(clauses))
        texts.append(" ".join(_sentence(clauses[i]) for i in order))
        gold.append(record)
    return ApplicationSample(texts, gold, attrs)
