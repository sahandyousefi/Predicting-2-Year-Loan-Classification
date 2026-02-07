import pytest

from credit_risk.llm.prompts import available_versions, load_prompt


@pytest.mark.parametrize("name", ["adverse_action", "underwriter_summary"])
def test_prompts_load_and_render(name):
    prompt = load_prompt(name)
    assert prompt.version == max(available_versions(name))
    values = {v: "x" for v in prompt.variables}
    system, user = prompt.render(**values)
    assert "$" not in user and system


def test_missing_variable_raises():
    with pytest.raises(KeyError):
        load_prompt("adverse_action").render(decision="decline")


def test_fingerprint_changes_with_text():
    a = load_prompt("adverse_action")
    b = load_prompt("underwriter_summary")
    assert a.fingerprint != b.fingerprint and len(a.fingerprint) == 12


def test_unknown_prompt():
    with pytest.raises(FileNotFoundError):
        load_prompt("nope")
