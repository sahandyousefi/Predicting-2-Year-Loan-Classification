"""Rule-based extractor used as a baseline and as the offline mock.

The rules classify each clause of an application by keywords and read the
number it states. They return the same JSON structure as the LLM, so the
validation and evaluation code is shared. The rules are a reference point for
the LLM, not a production extractor: they handle the phrasings they were
written for and miss paraphrases, negations and numbers written out in full
beyond twelve.
"""

from __future__ import annotations

import json
import re
from typing import Any

from credit_risk.llm.extraction import FIELDS, WORD_NUMBERS, numbers_in

_SENTENCE = re.compile(r"[.;!?](?=\s|$)|\n")
_CLAUSE = re.compile(r",\s+|\s+and\s+|\s+but\s+")
_WORDS = "|".join(sorted(WORD_NUMBERS, key=len, reverse=True))
_COUNT = re.compile(
    rf"\b(\d+|{_WORDS})\s+(?:times?|occasions?)\b|\b(once|twice)\b", re.I
)
_AGE = re.compile(
    r"\b(\d{2})\s*(?:years?[ -]old|yo)\b|\bi(?: am|'m)\s+(\d{2})\b|\baged?\s+(\d{2})\b",
    re.I,
)

PURPOSE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "debt_consolidation": ("consolidat", "high interest", "combine my"),
    "home_improvement": ("renovat", "roof", "basement", "home improvement"),
    "car": ("car ", "car,", "vehicle", "auto loan"),
    "education": ("tuition", "college", "degree", "course", "classes"),
    "major_purchase": ("furniture", "appliance", "piano", "computer equipment"),
    "wedding": ("wedding",),
    "moving": ("moving", "relocat", "first month rent"),
    "small_business": ("business", "online shop", "food truck"),
    "medical": ("medical", "hospital", "dental", "surgery"),
    "overdue_bills": ("past due", "collections", "behind on", "overdue"),
    "emergency": ("emergency", "urgent", "money fast"),
}
_PURPOSE_CUE = re.compile(
    r"\b(loan is for|loan for|money for|funds for|borrow|purpose|need this)\b", re.I
)


def _first_amount(clause: str) -> float | None:
    values = [v for v, u in numbers_in(clause) if u != "%" and v >= 10]
    return values[0] if values else None


def _count(clause: str) -> float | None:
    m = _COUNT.search(clause)
    if m:
        token = (m.group(1) or m.group(2)).lower()
        return float(WORD_NUMBERS[token]) if token in WORD_NUMBERS else float(token)
    if re.search(r"\bnever\b", clause, re.I):
        return 0.0
    return None


def _small_number(clause: str) -> float | None:
    if re.search(r"\b(no|zero|none)\b", clause, re.I):
        return 0.0
    values = [v for v, u in numbers_in(clause) if u == "" and v < 100]
    return values[0] if values else None


def _late_bucket(clause: str) -> str | None:
    if re.search(r"\b90\b", clause):
        return "late_90"
    if re.search(r"\b60\b", clause):
        return "late_60_89"
    if re.search(r"\b30\b", clause):
        return "late_30_59"
    return None


def rule_based_extract(text: str) -> dict[str, Any]:
    """Return ``{"fields": {name: {"value", "evidence"}}}`` for ``text``."""
    found: dict[str, tuple[Any, str]] = {}

    def put(name: str, value: Any, evidence: str) -> None:
        if value is not None and name not in found:
            found[name] = (value, evidence.strip())

    for sentence in filter(None, (s.strip() for s in _SENTENCE.split(text))):
        lower = sentence.lower()
        if _PURPOSE_CUE.search(sentence):
            for purpose, keys in PURPOSE_KEYWORDS.items():
                if any(k in lower + " " for k in keys):
                    put("purpose", purpose, sentence)
                    break
            continue
        for clause in filter(None, (c.strip() for c in _CLAUSE.split(sentence))):
            low = clause.lower()
            if re.search(r"late|past due|delinquen", low):
                bucket = _late_bucket(clause)
                count = _count(clause)
                if bucket:
                    put(bucket, count, clause)
                elif count == 0:
                    for name in ("late_30_59", "late_60_89", "late_90"):
                        put(name, 0.0, clause)
            elif "%" in clause or "percent" in low:
                pct = [v for v, u in numbers_in(clause) if u == "%"]
                if pct and re.search(r"limit|utiliz|card|balance|revolving", low):
                    put("rev_util", round(pct[0] / 100, 4), clause)
            elif re.search(r"\b(dependents?|kids?|child|children)\b", low):
                put("dependents", _small_number(clause), clause)
            elif re.search(r"mortgage|real estate|propert", low):
                put("real_estate", _small_number(clause), clause)
            elif re.search(r"\b(open|active)\b", low) and re.search(
                r"account|line|credit|card", low
            ):
                put("open_credit", _small_number(clause), clause)
            elif re.search(r"payment|\bpay\b", low) and re.search(
                r"debt|loan|card|toward", low
            ):
                put("monthly_debt", _first_amount(clause), clause)
            elif re.search(r"earn|income|take[- ]home|salary|\bmake\b", low):
                put("monthly_inc", _first_amount(clause), clause)
            m = _AGE.search(clause)
            if m:
                put("age", float(next(g for g in m.groups() if g)), m.group(0))
    fields = {}
    for spec in FIELDS:
        value, evidence = found.get(spec.name, (None, None))
        fields[spec.name] = {"value": value, "evidence": evidence}
    return {"fields": fields}


def application_from_prompt(user: str) -> str:
    """Application text between the triple quotes of the extraction prompt."""
    m = re.search(r'"""\s*(.*?)\s*"""', user, re.S)
    if not m:
        raise ValueError("no application text in prompt")
    return m.group(1)


def rule_based_responder(system: str, user: str) -> str:
    """``MockLLM`` responder that answers with the rule-based extraction."""
    return json.dumps(rule_based_extract(application_from_prompt(user)))
