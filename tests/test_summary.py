import json
import re

import pytest

from credit_risk.explain.reason_codes import REASON_TEXT, ReasonCode
from credit_risk.llm.client import MockLLM
from credit_risk.llm.retrieval import TfidfRetriever, load_policy_corpus
from credit_risk.llm.summary import retrieve_policy, underwriter_summary

REASONS = [
    ReasonCode("DELINQ", REASON_TEXT["DELINQ"], 1.1),
    ReasonCode("UTIL", REASON_TEXT["UTIL"], 0.5),
]


@pytest.fixture(scope="module")
def retriever():
    return TfidfRetriever(load_policy_corpus())


def _cite_first_shown(system, user):
    first = re.search(r"\[(POL-\d+)\]", user).group(1)
    return json.dumps(
        {
            "summary": f"Past due history drives the decline [{first}].",
            "policy_citations": [first],
            "recommendation": "decline",
        }
    )


def test_retrieves_relevant_policy(retriever):
    ids = [c.id for c in retrieve_policy(retriever, "decline", REASONS)]
    assert "POL-3" in ids and "POL-4" in ids


def test_valid_summary(retriever):
    result = underwriter_summary(
        0.71, "decline", REASONS, retriever, MockLLM(_cite_first_shown)
    )
    assert result.valid, result.errors


def test_citation_outside_context_is_flagged(retriever):
    reply = json.dumps(
        {
            "summary": "See [POL-99].",
            "policy_citations": ["POL-99"],
            "recommendation": "refer",
        }
    )
    result = underwriter_summary(
        0.5, "decline", REASONS, retriever, MockLLM(lambda s, u: reply)
    )
    assert not result.valid
    assert "POL-99" in result.errors[0]


def test_bad_recommendation_flagged(retriever):
    reply = json.dumps(
        {"summary": "ok", "policy_citations": [], "recommendation": "maybe"}
    )
    result = underwriter_summary(
        0.5, "decline", REASONS, retriever, MockLLM(lambda s, u: reply)
    )
    assert not result.valid
