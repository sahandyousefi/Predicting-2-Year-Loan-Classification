from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from credit_risk.explain.reason_codes import ReasonCode
from credit_risk.llm.client import LLMClient
from credit_risk.llm.prompts import load_prompt

# Prohibited bases under ECOA plus terms that would reveal them.
PROHIBITED = re.compile(
    r"\b(age|aged|old|young|elderly|race|racial|ethnic\w*|sex|gender|marital|married|"
    r"single|religio\w*|national origin|public assistance|pregnan\w*)\b",
    re.I,
)
NEXT_STEPS_FALLBACK = (
    "You may request the specific reasons for this decision in writing within "
    "60 days of this notice."
)


@dataclass
class NoticeResult:
    notice: dict[str, Any]
    valid: bool
    errors: list[str] = field(default_factory=list)
    fallback_used: bool = False
    attempts: int = 0
    prompt_fingerprint: str = ""


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value must be an object")
    return obj


def validate_notice(notice: dict[str, Any], reasons: list[ReasonCode]) -> list[str]:
    """Schema, grounding and compliance checks. Returns a list of problems."""
    errors = []
    for key in ("summary", "reasons", "next_steps"):
        if key not in notice:
            errors.append(f"missing key '{key}'")
    if errors:
        return errors

    items = notice["reasons"]
    if not isinstance(items, list) or not all(
        isinstance(r, dict) and {"code", "statement"} <= set(r) for r in items
    ):
        return ["'reasons' must be a list of objects with code and statement"]

    expected = [r.code for r in reasons]
    got = [r["code"] for r in items]
    unknown = [c for c in got if c not in expected]
    if unknown:
        errors.append(f"codes not in model output: {unknown}")
    if got != expected and not unknown:
        errors.append(f"codes must be {expected} in that order, got {got}")

    text = " ".join(
        [str(notice["summary"]), str(notice["next_steps"])]
        + [str(r["statement"]) for r in items]
    )
    hit = PROHIBITED.search(text)
    if hit:
        errors.append(f"prohibited term '{hit.group(0)}'")
    return errors


def template_notice(reasons: list[ReasonCode]) -> dict[str, Any]:
    """Deterministic notice used when the model output fails validation."""
    return {
        "summary": "We are unable to approve your application at this time.",
        "reasons": [
            {"code": r.code, "statement": r.description + "."} for r in reasons
        ],
        "next_steps": NEXT_STEPS_FALLBACK,
    }


def generate_notice(
    reasons: list[ReasonCode],
    client: LLMClient,
    decision: str = "decline",
    max_attempts: int = 2,
    version: int | None = None,
) -> NoticeResult:
    """Draft a notice with the LLM and accept it only if it passes validation.

    Every reason must come from the model's reason codes, in order, and the
    text must not reference a prohibited basis. Otherwise the template notice
    is returned, so a customer never receives unverified text.

    ``version`` selects a prompt version; by default the production one.
    """
    prompt = load_prompt("adverse_action", version)
    listed = "\n".join(f"- {r.code}: {r.description}" for r in reasons)
    system, user = prompt.render(decision=decision, reasons=listed)
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        response = client.complete(system, user, temperature=0.0, json_mode=True)
        try:
            notice = parse_json(response.text)
        except (json.JSONDecodeError, ValueError) as exc:
            errors = [f"invalid JSON: {exc}"]
            continue
        errors = validate_notice(notice, reasons)
        if not errors:
            return NoticeResult(notice, True, [], False, attempt, prompt.fingerprint)
    return NoticeResult(
        template_notice(reasons), False, errors, True, max_attempts, prompt.fingerprint
    )
