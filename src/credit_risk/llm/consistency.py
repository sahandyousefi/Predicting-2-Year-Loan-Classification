"""Consistency checks between extracted application fields and bureau data.

An extracted value is what the applicant wrote, which may be wrong because
the extractor misread it or because the applicant misstated it. Where the
credit bureau holds the same attribute, the two are compared field by field.
A difference beyond the field's tolerance is classified by direction, and a
difference in the direction that lowers the applicant's apparent risk
(income higher, debt, utilization or delinquencies lower than on file, or a
different age) is material and sends the application to review.

The bureau is the system of record: model inputs take the bureau value when
there is one and the stated value only to fill a gap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Check:
    """Comparison rule for one field.

    ``tolerance`` is relative to the bureau value when ``relative`` is true and
    absolute otherwise. ``adverse`` is the direction of a stated value that
    understates risk: ``+1`` if stated above the bureau, ``-1`` if below, and
    ``0`` if any difference is material.
    """

    name: str
    tolerance: float
    relative: bool
    adverse: int | None
    code: str


CHECKS: tuple[Check, ...] = (
    Check("age", 1.0, False, 0, "AGE_MISMATCH"),
    Check("monthly_inc", 0.10, True, +1, "INCOME_OVERSTATED"),
    Check("monthly_debt", 0.20, True, -1, "DEBT_UNDERSTATED"),
    Check("rev_util", 0.10, False, -1, "UTILIZATION_UNDERSTATED"),
    Check("late_30_59", 0.0, False, -1, "DELINQUENCY_UNDERSTATED"),
    Check("late_60_89", 0.0, False, -1, "DELINQUENCY_UNDERSTATED"),
    Check("late_90", 0.0, False, -1, "DELINQUENCY_UNDERSTATED"),
    Check("open_credit", 0.0, False, None, "OPEN_CREDIT_DIFFERS"),
    Check("real_estate", 0.0, False, None, "REAL_ESTATE_DIFFERS"),
    Check("dependents", 0.0, False, None, "DEPENDENTS_DIFFER"),
)

BUREAU_FIELDS = (
    "age",
    "monthly_inc",
    "rev_util",
    "late_30_59",
    "late_60_89",
    "late_90",
    "open_credit",
    "real_estate",
    "dependents",
    "debt_ratio",
)


@dataclass
class ConsistencyReport:
    """Field comparisons, material findings and reconciled model inputs."""

    rows: list[dict]
    material: list[str] = field(default_factory=list)
    inputs: dict[str, float | None] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return bool(self.material)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def bureau_reference(bureau: Mapping[str, float]) -> dict[str, float | None]:
    """Bureau values on the scale of the extracted fields.

    The bureau stores the debt ratio; monthly debt payments follow from it and
    the income on file.
    """
    ref = {k: _number(bureau.get(k)) for k in BUREAU_FIELDS if k != "debt_ratio"}
    ratio, inc = _number(bureau.get("debt_ratio")), ref.get("monthly_inc")
    ref["monthly_debt"] = ratio * inc if ratio is not None and inc else None
    return ref


def _number(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else x


def compare_field(check: Check, stated: float | None, ref: float | None) -> dict:
    row = {
        "field": check.name,
        "stated": stated,
        "bureau": ref,
        "difference": None,
        "status": "consistent",
        "material": False,
    }
    if stated is None:
        row["status"] = "not_stated"
        return row
    if ref is None:
        row["status"] = "no_reference"
        return row
    diff = stated - ref
    row["difference"] = diff
    scale = abs(ref) if check.relative else 1.0
    if check.relative and ref == 0:
        within = stated == 0
    else:
        within = abs(diff) <= check.tolerance * scale + 1e-9
    if within:
        return row
    direction = 1 if diff > 0 else -1
    row["status"] = "overstated" if direction > 0 else "understated"
    row["material"] = check.adverse == 0 or check.adverse == direction
    return row


def reconcile(
    extracted: Mapping[str, float | str | None],
    bureau: Mapping[str, float],
    checks: Sequence[Check] = CHECKS,
) -> ConsistencyReport:
    """Compare extracted fields with the bureau record and build model inputs.

    Parameters
    ----------
    extracted : mapping
        Accepted values from :func:`credit_risk.llm.extraction.extract_fields`.
    bureau : mapping
        Bureau attributes with the benchmark names, including ``debt_ratio``.
    """
    ref = bureau_reference(bureau)
    rows, material = [], []
    for check in checks:
        stated = extracted.get(check.name)
        row = compare_field(
            check, None if stated is None else float(stated), ref.get(check.name)
        )
        row["code"] = (
            check.code if row["status"] in {"overstated", "understated"} else ""
        )
        rows.append(row)
        if row["material"] and check.code not in material:
            material.append(check.code)

    inputs: dict[str, float | None] = {}
    for name in BUREAU_FIELDS:
        if name == "debt_ratio":
            continue
        stated = extracted.get(name)
        inputs[name] = ref[name] if ref[name] is not None else _number(stated)
    if bureau.get("debt_ratio") is not None:
        inputs["debt_ratio"] = _number(bureau["debt_ratio"])
    else:
        inc, debt = inputs.get("monthly_inc"), _number(extracted.get("monthly_debt"))
        inputs["debt_ratio"] = debt / inc if debt is not None and inc else None
    return ConsistencyReport(rows, material, inputs)


def misstatement_study(
    reports: Sequence[ConsistencyReport], misstated: Sequence[str | None]
) -> pd.DataFrame:
    """Detection rate by injected misstatement type and false alarms.

    The ``honest`` row gives the share of applications without a misstatement
    that were flagged; the other rows give the share flagged with the matching
    reason code.
    """
    expected = {
        "income_overstated": "INCOME_OVERSTATED",
        "debt_understated": "DEBT_UNDERSTATED",
        "utilization_understated": "UTILIZATION_UNDERSTATED",
        "delinquency_understated": "DELINQUENCY_UNDERSTATED",
    }
    rows = []
    for kind in [None, *expected]:
        idx = [i for i, m in enumerate(misstated) if m == kind]
        if not idx:
            continue
        if kind is None:
            hits = [reports[i].needs_review for i in idx]
        else:
            hits = [expected[kind] in reports[i].material for i in idx]
        rows.append(
            {
                "misstatement": kind or "honest",
                "applications": len(idx),
                "flagged": float(np.mean(hits)),
            }
        )
    return pd.DataFrame(rows)
