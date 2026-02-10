"""Structured field extraction from free-text loan applications.

The model returns, for each field of a fixed schema, a value and an exact
quote from the application that supports it. A field is accepted only if

1. the value has the declared type and lies within its bounds,
2. the quote occurs in the application text, and
3. for numeric fields, the quote contains a number equal to the value after
   unit handling (``4.2k``, ``$4,200``, ``45%``, ``twice``).

Fields that fail are set to null and listed, and the application goes to
manual review. A value can therefore never enter the credit model unless the
applicant wrote it. Text flagged by the injection screen is not sent to the
model at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from credit_risk.data.synthetic_text import PURPOSES
from credit_risk.llm.client import LLMClient
from credit_risk.llm.guardrails import screen_user_text
from credit_risk.llm.notices import parse_json
from credit_risk.llm.prompts import load_prompt

MAX_CHARS = 2000


@dataclass(frozen=True)
class FieldSpec:
    """One extractable field.

    ``kind`` is ``"int"``, ``"amount"`` (currency per month), ``"percent"``
    (returned as a fraction) or ``"category"``.
    """

    name: str
    kind: str
    description: str
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("age", "int", "applicant age in years", 18, 110),
    FieldSpec("monthly_inc", "amount", "monthly income in dollars", 0, 1e6),
    FieldSpec(
        "monthly_debt", "amount", "monthly payments on existing debt in dollars", 0, 1e6
    ),
    FieldSpec("dependents", "int", "number of dependents", 0, 20),
    FieldSpec("open_credit", "int", "number of open credit lines and loans", 0, 100),
    FieldSpec("real_estate", "int", "number of mortgages or real estate loans", 0, 50),
    FieldSpec(
        "rev_util",
        "percent",
        "share of revolving credit limits in use, as a fraction (45% is 0.45)",
        0,
        5,
    ),
    FieldSpec("late_30_59", "int", "times 30 to 59 days past due", 0, 100),
    FieldSpec("late_60_89", "int", "times 60 to 89 days past due", 0, 100),
    FieldSpec("late_90", "int", "times 90 or more days past due", 0, 100),
    FieldSpec(
        "purpose",
        "category",
        "loan purpose, one of: " + ", ".join(PURPOSES),
        choices=tuple(PURPOSES),
    ),
)
FIELD_BY_NAME = {f.name: f for f in FIELDS}

WORD_NUMBERS = {
    "no": 0,
    "none": 0,
    "zero": 0,
    "never": 0,
    "one": 1,
    "once": 1,
    "a single": 1,
    "two": 2,
    "twice": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_WORDS = "|".join(sorted(WORD_NUMBERS, key=len, reverse=True))
_NUMBER = re.compile(
    r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)\s*(k\b|thousand\b|%|percent\b)?"
    rf"|\b({_WORDS})\b",
    re.I,
)


@dataclass
class ExtractionResult:
    values: dict[str, Any]
    evidence: dict[str, str | None]
    rejected: dict[str, list[str]] = field(default_factory=dict)
    blocked: bool = False
    flags: list[str] = field(default_factory=list)
    attempts: int = 0
    prompt_fingerprint: str = ""

    @property
    def needs_review(self) -> bool:
        return self.blocked or bool(self.rejected)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def numbers_in(text: str) -> list[tuple[float, str]]:
    """Numbers in ``text`` with their unit: ``""``, ``"k"`` or ``"%"``."""
    found = []
    for m in _NUMBER.finditer(text):
        if m.group(3):
            found.append((float(WORD_NUMBERS[m.group(3).lower()]), ""))
            continue
        value = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit in {"k", "thousand"}:
            found.append((value * 1000, "k"))
        elif unit in {"%", "percent"}:
            found.append((value, "%"))
        else:
            found.append((value, ""))
    return found


def supported(spec: FieldSpec, value: float, evidence: str) -> bool:
    """Whether ``evidence`` states ``value`` for a numeric field."""
    candidates = numbers_in(evidence)
    if spec.kind == "percent":
        return any(abs(v / 100 - value) < 5e-3 for v, u in candidates if u == "%")
    tol = 0.5 if spec.kind == "amount" else 1e-9
    return any(abs(v - value) <= tol for v, u in candidates if u != "%")


def validate_field(spec: FieldSpec, item: Any, text: str) -> tuple[Any, list[str]]:
    """Check one extracted field against the schema and the source text.

    Returns the accepted value (``None`` if rejected or absent) and the list
    of problems.
    """
    if not isinstance(item, dict) or not {"value", "evidence"} <= set(item):
        return None, ["expected an object with value and evidence"]
    value, evidence = item["value"], item["evidence"]
    if value is None:
        return None, []
    if not isinstance(evidence, str) or not evidence.strip():
        return None, ["value given without evidence"]
    if _normalize(evidence) not in _normalize(text):
        return None, ["evidence is not a quote from the application"]
    if spec.kind == "category":
        if value not in spec.choices:
            return None, [f"'{value}' is not an allowed category"]
        return value, []
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, ["value must be a number"]
    if spec.kind == "int" and float(value) != int(value):
        return None, ["value must be a whole number"]
    if (spec.lo is not None and value < spec.lo) or (
        spec.hi is not None and value > spec.hi
    ):
        return None, [f"value {value} outside [{spec.lo}, {spec.hi}]"]
    if not supported(spec, float(value), evidence):
        return None, ["evidence does not state the value"]
    return (int(value) if spec.kind == "int" else float(value)), []


def validate_extraction(
    output: dict[str, Any], text: str, fields: tuple[FieldSpec, ...] = FIELDS
) -> tuple[dict[str, Any], dict[str, str | None], dict[str, list[str]]]:
    """Validate every field; unknown or missing fields are reported."""
    items = output.get("fields")
    if not isinstance(items, dict):
        raise ValueError("'fields' must be an object")
    values: dict[str, Any] = {}
    evidence: dict[str, str | None] = {}
    rejected: dict[str, list[str]] = {}
    for spec in fields:
        if spec.name not in items:
            values[spec.name], evidence[spec.name] = None, None
            rejected[spec.name] = ["field missing from output"]
            continue
        value, errors = validate_field(spec, items[spec.name], text)
        values[spec.name] = value
        evidence[spec.name] = (
            items[spec.name]["evidence"] if value is not None else None
        )
        if errors:
            rejected[spec.name] = errors
    unknown = sorted(set(items) - {f.name for f in fields})
    if unknown:
        rejected["_unknown"] = [f"fields not in the schema: {unknown}"]
    return values, evidence, rejected


def field_list(fields: tuple[FieldSpec, ...] = FIELDS) -> str:
    return "\n".join(f"- {f.name}: {f.description}" for f in fields)


def extract_fields(
    text: str,
    client: LLMClient,
    fields: tuple[FieldSpec, ...] = FIELDS,
    max_attempts: int = 2,
    version: int | None = None,
) -> ExtractionResult:
    """Extract and validate structured fields from one application.

    The request is retried only when the reply is not valid JSON with a
    ``fields`` object. Field-level failures are not retried; they are
    returned in ``rejected`` for review.

    ``version`` selects a prompt version; by default the production one.
    """
    prompt = load_prompt("application_extraction", version)
    screen = screen_user_text(text, max_chars=MAX_CHARS)
    empty = {f.name: None for f in fields}
    if screen.blocked:
        return ExtractionResult(
            dict(empty),
            dict(empty),
            blocked=True,
            flags=screen.flags,
            prompt_fingerprint=prompt.fingerprint,
        )
    system, user = prompt.render(fields=field_list(fields), text=screen.text)
    error = ""
    for attempt in range(1, max_attempts + 1):
        response = client.complete(system, user, temperature=0.0, json_mode=True)
        try:
            values, evidence, rejected = validate_extraction(
                parse_json(response.text), screen.text, fields
            )
        except (json.JSONDecodeError, ValueError) as exc:
            error = f"invalid output: {exc}"
            continue
        return ExtractionResult(
            values,
            evidence,
            rejected,
            attempts=attempt,
            prompt_fingerprint=prompt.fingerprint,
        )
    return ExtractionResult(
        dict(empty),
        dict(empty),
        {"_output": [error]},
        attempts=max_attempts,
        prompt_fingerprint=prompt.fingerprint,
    )


def to_model_inputs(values: dict[str, Any]) -> dict[str, float | None]:
    """Map extracted fields to the benchmark attributes.

    The debt ratio is monthly debt payments over monthly income and is left
    missing if either is missing or income is zero.
    """
    out = {
        k: values.get(k)
        for k in (
            "age",
            "monthly_inc",
            "dependents",
            "open_credit",
            "real_estate",
            "rev_util",
            "late_30_59",
            "late_60_89",
            "late_90",
        )
    }
    inc, debt = values.get("monthly_inc"), values.get("monthly_debt")
    out["debt_ratio"] = debt / inc if inc and debt is not None else None
    return out
