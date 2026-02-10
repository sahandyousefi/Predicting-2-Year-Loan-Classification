import json

import numpy as np
import pytest

from credit_risk.cli import main
from credit_risk.data.synthetic_applications import make_applications
from credit_risk.llm.client import MockLLM
from credit_risk.llm.extraction_baseline import rule_based_responder
from credit_risk.llm.prompt_report import (
    compare_prompt_versions,
    echo_notice_responder,
    mcnemar_exact,
)


def test_mcnemar_exact_values():
    a = np.array([True] * 10)
    assert mcnemar_exact(a, a) == 1.0
    # Eight discordant pairs all favouring one version: 2 * 0.5**8.
    b = np.array([False] * 8 + [True] * 2)
    assert mcnemar_exact(a, b) == pytest.approx(2 * 0.5**8)
    assert mcnemar_exact(b, a) == pytest.approx(2 * 0.5**8)
    with pytest.raises(ValueError):
        mcnemar_exact(a, b[:5])


def _version_sensitive_notices(system, user):
    reply = json.loads(echo_notice_responder(system, user))
    if "25 words" not in system:
        # The v1 prompt in this mock invents an extra reason.
        reply["reasons"].append({"code": "EMPLOYMENT", "statement": "Job history."})
    return json.dumps(reply)


def test_notice_report_detects_a_better_version():
    table = compare_prompt_versions(
        "notices", MockLLM(_version_sensitive_notices), [1, 2]
    ).set_index("version")
    assert table.loc[1, "success_rate"] == 0.0
    assert table.loc[2, "success_rate"] == 1.0
    assert table.loc[2, "status"] == "candidate"
    assert table.loc[2, "mcnemar_p_vs_baseline"] < 0.01
    assert np.isnan(table.loc[1, "mcnemar_p_vs_baseline"])
    assert table.loc[1, "fingerprint"] != table.loc[2, "fingerprint"]


def test_extraction_report_with_identical_behaviour():
    sample = make_applications(30, seed=3)
    table = compare_prompt_versions(
        "extraction",
        MockLLM(rule_based_responder),
        [1, 2],
        cases=sample.texts,
        gold=sample.gold,
    ).set_index("version")
    assert table.loc[1, "field_accuracy"] == table.loc[2, "field_accuracy"]
    assert table.loc[2, "mcnemar_p_vs_baseline"] == 1.0
    # The candidate adds rules and an example, which costs prompt tokens.
    assert (
        table.loc[2, "prompt_tokens_per_call"] > table.loc[1, "prompt_tokens_per_call"]
    )
    assert (table["calls"] == 30).all()


def test_unknown_task():
    with pytest.raises(ValueError):
        compare_prompt_versions("summary", MockLLM(), [1])


@pytest.mark.parametrize("task", ["notices", "extraction"])
def test_cli_prompt_report(task, capsys):
    assert main(["prompt-report", "--task", task, "--versions", "1", "2"]) == 0
    out = capsys.readouterr().out
    assert "candidate" in out and "mcnemar_p_vs_baseline" in out
