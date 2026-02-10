import pytest

from credit_risk.llm.prompts import (
    available_versions,
    load_prompt,
    production_version,
)

NAMES = ["adverse_action", "underwriter_summary", "application_extraction"]


@pytest.mark.parametrize("name", NAMES)
def test_prompts_load_and_render(name):
    prompt = load_prompt(name)
    assert prompt.version == production_version(name)
    assert prompt.status == "production"
    values = {v: "x" for v in prompt.variables}
    system, user = prompt.render(**values)
    assert "$" not in user and system


@pytest.mark.parametrize("name", NAMES)
def test_every_version_renders_with_the_same_variables(name):
    versions = [load_prompt(name, v) for v in available_versions(name)]
    assert all(p.variables == versions[0].variables for p in versions)


def test_candidates_load_only_when_requested():
    assert max(available_versions("adverse_action")) == 2
    assert load_prompt("adverse_action").version == 1
    candidate = load_prompt("adverse_action", 2)
    assert candidate.status == "candidate"
    assert candidate.fingerprint != load_prompt("adverse_action").fingerprint


def test_unknown_version():
    with pytest.raises(FileNotFoundError):
        load_prompt("adverse_action", 99)


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
