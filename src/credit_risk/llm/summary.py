from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from credit_risk.explain.reason_codes import ReasonCode
from credit_risk.llm.client import LLMClient
from credit_risk.llm.notices import parse_json
from credit_risk.llm.prompts import load_prompt
from credit_risk.llm.retrieval import Chunk, Retriever

RECOMMENDATIONS = {"approve", "decline", "refer"}


@dataclass
class CaseSummary:
    output: dict[str, Any] | None
    retrieved: list[str]
    valid: bool
    errors: list[str] = field(default_factory=list)
    prompt_fingerprint: str = ""


def build_query(decision: str, reasons: list[ReasonCode]) -> str:
    parts = [f"{decision} decision cut-off"] + [r.description for r in reasons]
    return ". ".join(parts)


def retrieve_policy(
    retriever: Retriever, decision: str, reasons: list[ReasonCode], k: int = 4
) -> list[Chunk]:
    """Union of the top hits for the overall case and for each reason."""
    seen: dict[str, Chunk] = {}
    for query in [build_query(decision, reasons)] + [r.description for r in reasons]:
        for chunk, _ in retriever.search(query, k=2):
            seen.setdefault(chunk.id, chunk)
    return list(seen.values())[: max(k, len(reasons))]


def validate_summary(output: dict[str, Any], retrieved_ids: list[str]) -> list[str]:
    errors = []
    for key in ("summary", "policy_citations", "recommendation"):
        if key not in output:
            errors.append(f"missing key '{key}'")
    if errors:
        return errors
    allowed = set(retrieved_ids)
    cited = set(output["policy_citations"]) | set(
        re.findall(r"\[([A-Z]+-\d+)\]", str(output["summary"]))
    )
    unsupported = sorted(cited - allowed)
    if unsupported:
        errors.append(f"citations not in retrieved excerpts: {unsupported}")
    if str(output["recommendation"]).lower() not in RECOMMENDATIONS:
        errors.append(f"recommendation must be one of {sorted(RECOMMENDATIONS)}")
    return errors


def underwriter_summary(
    pd_hat: float,
    decision: str,
    reasons: list[ReasonCode],
    retriever: Retriever,
    client: LLMClient,
) -> CaseSummary:
    """Retrieval-augmented case summary for an analyst.

    Citations are checked against the excerpts actually shown to the model,
    so an invalid result signals an unsupported statement rather than being
    passed on silently.
    """
    chunks = retrieve_policy(retriever, decision, reasons)
    ids = [c.id for c in chunks]
    prompt = load_prompt("underwriter_summary")
    system, user = prompt.render(
        pd=f"{pd_hat:.3f}",
        decision=decision,
        reasons="; ".join(f"{r.code} ({r.description})" for r in reasons) or "none",
        policy="\n".join(f"[{c.id}] {c.title}: {c.text}" for c in chunks),
    )
    response = client.complete(system, user, temperature=0.0, json_mode=True)
    try:
        output = parse_json(response.text)
    except (json.JSONDecodeError, ValueError) as exc:
        return CaseSummary(
            None, ids, False, [f"invalid JSON: {exc}"], prompt.fingerprint
        )
    errors = validate_summary(output, ids)
    return CaseSummary(output, ids, not errors, errors, prompt.fingerprint)
