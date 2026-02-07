import json

import pytest

from credit_risk.explain.reason_codes import REASON_TEXT, ReasonCode
from credit_risk.llm.client import MockLLM
from credit_risk.llm.notices import generate_notice, parse_json, validate_notice

REASONS = [
    ReasonCode("DELINQ", REASON_TEXT["DELINQ"], 0.9),
    ReasonCode("UTIL", REASON_TEXT["UTIL"], 0.4),
]


def _reply(codes, extra=""):
    return json.dumps(
        {
            "summary": "Your application was not approved." + extra,
            "reasons": [{"code": c, "statement": f"Statement for {c}."} for c in codes],
            "next_steps": "Contact us within 60 days to request details.",
        }
    )


def test_grounded_reply_accepted():
    result = generate_notice(REASONS, MockLLM(lambda s, u: _reply(["DELINQ", "UTIL"])))
    assert result.valid and not result.fallback_used and result.attempts == 1


def test_hallucinated_reason_falls_back():
    llm = MockLLM(lambda s, u: _reply(["DELINQ", "UTIL", "EMPLOYMENT"]))
    result = generate_notice(REASONS, llm)
    assert result.fallback_used
    assert [r["code"] for r in result.notice["reasons"]] == ["DELINQ", "UTIL"]
    assert any("EMPLOYMENT" in e for e in result.errors)


def test_prohibited_basis_rejected():
    reply = _reply(["DELINQ", "UTIL"], " Applicants of your age are higher risk.")
    assert any("prohibited" in e for e in validate_notice(json.loads(reply), REASONS))


def test_recovers_on_second_attempt():
    replies = iter(["not json", _reply(["DELINQ", "UTIL"])])
    result = generate_notice(REASONS, MockLLM(lambda s, u: next(replies)))
    assert result.valid and result.attempts == 2


def test_prompt_lists_only_model_reasons():
    llm = MockLLM(lambda s, u: _reply(["DELINQ", "UTIL"]))
    generate_notice(REASONS, llm)
    _, user = llm.calls[0]
    assert "DELINQ" in user and "UTIL" in user and "INCOME" not in user


def test_parse_json_strips_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("[1, 2]")
