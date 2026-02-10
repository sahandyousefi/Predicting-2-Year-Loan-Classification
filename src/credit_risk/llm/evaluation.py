from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
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


def _matches(spec, predicted, truth) -> bool:
    if spec.kind == "category":
        return predicted == truth
    tol = {"amount": 0.5, "percent": 5e-3}.get(spec.kind, 1e-9)
    return abs(float(predicted) - float(truth)) <= tol


def evaluate_extraction(
    client: LLMClient,
    texts: list[str],
    gold: list[dict],
) -> tuple[pd.DataFrame, dict]:
    """Field-level accuracy of application extraction against gold values.

    For each field, over applications that state it: ``accuracy`` is the
    share extracted with the correct value, ``missed`` the share returned as
    null and ``wrong`` the share with an accepted but incorrect value. Over
    applications that do not state it, ``hallucinated`` is the share with an
    accepted value. ``rejected`` counts field-level validation failures.

    Returns the per-field table and a summary with micro-averaged accuracy,
    the share of applications extracted without any error, the review rate
    and token usage.
    """
    from credit_risk.llm.extraction import FIELDS, extract_fields

    wrapped = client if isinstance(client, RetryingClient) else RetryingClient(client)
    results = [extract_fields(text, wrapped) for text in texts]
    rows = []
    for spec in FIELDS:
        stated = correct = missed = wrong = absent = halluc = rejected = 0
        for res, truth in zip(results, gold):
            value, true_value = res.values.get(spec.name), truth.get(spec.name)
            rejected += spec.name in res.rejected
            if true_value is None:
                absent += 1
                halluc += value is not None
            else:
                stated += 1
                if value is None:
                    missed += 1
                elif _matches(spec, value, true_value):
                    correct += 1
                else:
                    wrong += 1
        rows.append(
            {
                "field": spec.name,
                "stated": stated,
                "accuracy": correct / stated if stated else float("nan"),
                "missed": missed / stated if stated else float("nan"),
                "wrong": wrong / stated if stated else float("nan"),
                "hallucinated": halluc / absent if absent else float("nan"),
                "rejected": rejected,
            }
        )
    table = pd.DataFrame(rows)

    def perfect(res, truth) -> bool:
        for spec in FIELDS:
            value, true_value = res.values.get(spec.name), truth.get(spec.name)
            if (value is None) != (true_value is None):
                return False
            if value is not None and not _matches(spec, value, true_value):
                return False
        return True

    stated_total = int(table["stated"].sum())
    summary = {
        "n_applications": len(texts),
        "field_accuracy": round(
            float((table["accuracy"] * table["stated"]).sum() / stated_total), 4
        ),
        "hallucination_rate": round(
            float(
                np.nansum(table["hallucinated"] * (len(texts) - table["stated"]))
                / max(len(texts) * len(FIELDS) - stated_total, 1)
            ),
            4,
        ),
        "exact_application_rate": round(
            float(np.mean([perfect(r, g) for r, g in zip(results, gold)])), 4
        ),
        "review_rate": round(float(np.mean([r.needs_review for r in results])), 4),
        "usage": wrapped.usage.summary(),
    }
    return table, summary
