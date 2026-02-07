from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from credit_risk.explain.reason_codes import REASON_TEXT, ReasonCode
from credit_risk.llm.client import LLMClient, RetryingClient
from credit_risk.llm.guardrails import screen_user_text
from credit_risk.llm.notices import generate_notice

EVAL_DIR = Path(__file__).parent / "eval"


def load_notice_cases(
    path: Path = EVAL_DIR / "notice_cases.yaml",
) -> list[list[ReasonCode]]:
    cases = yaml.safe_load(path.read_text())
    return [
        [
            ReasonCode(code, REASON_TEXT[code], 1.0 / (i + 1))
            for i, code in enumerate(case)
        ]
        for case in cases
    ]


def evaluate_notices(
    client: LLMClient, cases: list[list[ReasonCode]] | None = None
) -> dict:
    """Run notice generation on a fixed case set and report quality rates.

    - format_compliance: first attempt parsed as JSON with the required keys
    - grounded_rate: accepted without fallback (codes match, no prohibited terms)
    - hallucination_rate: any attempt cited a code the model did not produce
    - fallback_rate: template notice used
    """
    cases = cases or load_notice_cases()
    wrapped = client if isinstance(client, RetryingClient) else RetryingClient(client)
    rows = []
    for reasons in cases:
        result = generate_notice(reasons, wrapped)
        rows.append(
            {
                "grounded": result.valid,
                "first_try": result.valid and result.attempts == 1,
                "hallucinated": any("not in model output" in e for e in result.errors),
                "prohibited": any("prohibited" in e for e in result.errors),
                "invalid_json": any("invalid JSON" in e for e in result.errors),
                "fallback": result.fallback_used,
            }
        )

    def rate(key: str) -> float:
        return round(float(np.mean([r[key] for r in rows])), 4)

    return {
        "n_cases": len(rows),
        "format_compliance": round(1 - rate("invalid_json"), 4),
        "grounded_rate": rate("grounded"),
        "first_try_rate": rate("first_try"),
        "hallucination_rate": rate("hallucinated"),
        "prohibited_term_rate": rate("prohibited"),
        "fallback_rate": rate("fallback"),
        "usage": wrapped.usage.summary(),
    }


def evaluate_guardrails(
    path: Path = EVAL_DIR / "injection_cases.yaml",
) -> dict[str, float]:
    """Detection rate on attacks and false positive rate on benign text."""
    data = yaml.safe_load(path.read_text())
    detected = [screen_user_text(t).blocked for t in data["attacks"]]
    false_pos = [screen_user_text(t).blocked for t in data["benign"]]
    return {
        "detection_rate": round(float(np.mean(detected)), 4),
        "false_positive_rate": round(float(np.mean(false_pos)), 4),
    }
