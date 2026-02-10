"""Side-by-side evaluation of prompt versions.

Every version runs on the same evaluation cases with the same client, so the
per-case outcomes are paired. The report gives the task metrics, token usage
and the template fingerprint of each version, and an exact McNemar test of
each version against the baseline on the per-case success indicator
(McNemar, 1947). Only the discordant cases, where one version succeeds and the
other fails, carry information about the difference.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from credit_risk.explain.reason_codes import ReasonCode
from credit_risk.llm.client import LLMClient, RetryingClient
from credit_risk.llm.prompts import load_prompt

TASKS = {"notices": "adverse_action", "extraction": "application_extraction"}


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("outcomes must be paired")
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    if only_a + only_b == 0:
        return 1.0
    return float(binomtest(only_a, only_a + only_b, 0.5).pvalue)


def notice_outcomes(
    client: LLMClient, cases: list[list[ReasonCode]], version: int
) -> pd.DataFrame:
    """Per-case results of notice generation with one prompt version."""
    from credit_risk.llm.notices import generate_notice

    rows = []
    for reasons in cases:
        result = generate_notice(reasons, client, version=version)
        words = [len(str(r["statement"]).split()) for r in result.notice["reasons"]]
        rows.append(
            {
                "success": result.valid,
                "first_try": result.valid and result.attempts == 1,
                "fallback": result.fallback_used,
                "max_statement_words": max(words) if result.valid else np.nan,
            }
        )
    return pd.DataFrame(rows)


def extraction_outcomes(
    client: LLMClient, texts: list[str], gold: list[dict], version: int
) -> pd.DataFrame:
    """Per-application results of field extraction with one prompt version."""
    from credit_risk.llm.evaluation import _matches
    from credit_risk.llm.extraction import FIELDS, extract_fields

    rows = []
    for text, truth in zip(texts, gold):
        result = extract_fields(text, client, version=version)
        correct = stated = 0
        for spec in FIELDS:
            value, true_value = result.values.get(spec.name), truth.get(spec.name)
            if true_value is not None:
                stated += 1
                correct += value is not None and _matches(spec, value, true_value)
        exact = all(
            (result.values.get(s.name) is None) == (truth.get(s.name) is None)
            and (
                truth.get(s.name) is None
                or _matches(s, result.values[s.name], truth[s.name])
            )
            for s in FIELDS
        )
        rows.append(
            {
                "success": exact,
                "field_accuracy": correct / stated if stated else np.nan,
                "review": result.needs_review,
            }
        )
    return pd.DataFrame(rows)


def compare_prompt_versions(
    task: str,
    client: LLMClient,
    versions: Sequence[int],
    cases: list | None = None,
    gold: list[dict] | None = None,
) -> pd.DataFrame:
    """Evaluate prompt versions of ``task`` on the same cases.

    Parameters
    ----------
    task : {"notices", "extraction"}
        Adverse action notices on reason code combinations, or field
        extraction on application texts with gold values.
    client : LLMClient
        Model under test. Each version gets its own usage log.
    versions : sequence of int
        Prompt versions; the first is the baseline of the McNemar tests.
    cases, gold
        Reason code lists for notices; texts and gold records for extraction.
        Defaults are the stored notice cases and 200 synthetic applications.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    if task == "notices" and cases is None:
        from credit_risk.llm.evaluation import load_notice_cases

        cases = load_notice_cases()
    if task == "extraction" and cases is None:
        from credit_risk.data.synthetic_applications import make_applications

        sample = make_applications(200, seed=0)
        cases, gold = sample.texts, sample.gold

    outcomes, rows = {}, []
    for version in versions:
        prompt = load_prompt(TASKS[task], version)
        wrapped = RetryingClient(client)
        if task == "notices":
            frame = notice_outcomes(wrapped, cases, version)
            metrics = {
                "success_rate": frame["success"].mean(),
                "first_try_rate": frame["first_try"].mean(),
                "max_statement_words": frame["max_statement_words"].max(),
            }
        else:
            frame = extraction_outcomes(wrapped, cases, gold, version)
            metrics = {
                "success_rate": frame["success"].mean(),
                "field_accuracy": frame["field_accuracy"].mean(),
                "review_rate": frame["review"].mean(),
            }
        usage = wrapped.usage.summary()
        outcomes[version] = frame["success"].to_numpy()
        rows.append(
            {
                "version": version,
                "status": prompt.status,
                "fingerprint": prompt.fingerprint,
                "cases": len(frame),
                **{k: float(v) for k, v in metrics.items()},
                "prompt_tokens_per_call": usage["prompt_tokens"] / usage["calls"],
                "calls": usage["calls"],
                "cost": usage["cost"],
            }
        )
    base = outcomes[versions[0]]
    for row in rows:
        row["mcnemar_p_vs_baseline"] = (
            np.nan
            if row["version"] == versions[0]
            else mcnemar_exact(base, outcomes[row["version"]])
        )
    return pd.DataFrame(rows)


def echo_notice_responder(system: str, user: str) -> str:
    """Offline notice responder that restates the listed reasons.

    It returns a well-formed, grounded notice for any prompt version, so it
    exercises the pipeline but cannot show differences between versions.
    """
    reasons = re.findall(r"^- ([A-Z_]+): (.+)$", user, re.M)
    return json.dumps(
        {
            "summary": "We are unable to approve your application at this time.",
            "reasons": [
                {"code": code, "statement": f"{text.strip()}."}
                for code, text in reasons
            ],
            "next_steps": "You may request the reasons for this decision in writing.",
        }
    )
