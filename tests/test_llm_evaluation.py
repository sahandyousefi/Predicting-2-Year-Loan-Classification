import json
import re

from credit_risk.llm.client import MockLLM
from credit_risk.llm.evaluation import (
    evaluate_guardrails,
    evaluate_notices,
    load_notice_cases,
)
from credit_risk.llm.guardrails import sanitize, screen_user_text


def _faithful(system, user):
    codes = re.findall(r"^- (\w+):", user, re.M)
    return json.dumps(
        {
            "summary": "Your application was not approved.",
            "reasons": [{"code": c, "statement": f"Reason {c}."} for c in codes],
            "next_steps": "Write to us within 60 days for details.",
        }
    )


def _adds_reason(system, user):
    obj = json.loads(_faithful(system, user))
    obj["reasons"].append({"code": "EMPLOYMENT", "statement": "Short job tenure."})
    return json.dumps(obj)


def test_faithful_model_scores_perfectly():
    report = evaluate_notices(MockLLM(_faithful))
    assert report["n_cases"] == len(load_notice_cases())
    assert report["grounded_rate"] == 1.0
    assert report["hallucination_rate"] == 0.0
    assert report["usage"]["calls"] == report["n_cases"]


def test_hallucinating_model_is_caught():
    report = evaluate_notices(MockLLM(_adds_reason))
    assert report["hallucination_rate"] == 1.0
    assert report["fallback_rate"] == 1.0


def test_non_json_model_lowers_format_compliance():
    report = evaluate_notices(MockLLM(lambda s, u: "Sorry, I cannot help."))
    assert report["format_compliance"] == 0.0


def test_guardrail_cases():
    metrics = evaluate_guardrails()
    assert metrics["detection_rate"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_sanitize_strips_control_and_caps():
    text = "loan\x00 for​ car " + "x" * 1000
    clean = sanitize(text, max_chars=20)
    assert "\x00" not in clean and len(clean) == 20
    assert not screen_user_text("Buy a used car for work.").blocked
