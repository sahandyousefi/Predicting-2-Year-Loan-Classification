import json

import pytest

from credit_risk.cli import main
from credit_risk.data.synthetic_applications import make_applications
from credit_risk.llm.client import MockLLM
from credit_risk.llm.evaluation import evaluate_extraction
from credit_risk.llm.extraction import FIELD_BY_NAME, FIELDS, numbers_in, supported
from credit_risk.llm.extraction_baseline import (
    application_from_prompt,
    rule_based_extract,
    rule_based_responder,
)


@pytest.fixture(scope="module")
def sample():
    return make_applications(80, seed=5)


def test_generator_is_reproducible(sample):
    again = make_applications(80, seed=5)
    assert again.texts == sample.texts
    assert again.gold == sample.gold
    assert make_applications(80, seed=6).texts != sample.texts


def test_gold_fields_and_missingness(sample):
    names = {f.name for f in FIELDS}
    assert all(set(g) == names for g in sample.gold)
    share = sum(g["age"] is None for g in sample.gold) / len(sample.gold)
    assert 0.0 < share < 0.4
    none_missing = make_applications(30, missing_rate=0.0, seed=1)
    assert all(v is not None for g in none_missing.gold for v in g.values())


def test_every_stated_value_appears_in_the_text(sample):
    for text, gold in zip(sample.texts, sample.gold):
        stated = numbers_in(text)
        for name, value in gold.items():
            spec = FIELD_BY_NAME[name]
            if value is None or spec.kind == "category":
                continue
            assert supported(spec, value, text), (name, value, text)
        assert stated


def test_rules_are_exact_on_covered_phrasings():
    easy = make_applications(60, hard_rate=0.0, seed=2)
    table, summary = evaluate_extraction(
        MockLLM(rule_based_responder), easy.texts, easy.gold
    )
    assert summary["field_accuracy"] == 1.0
    assert summary["hallucination_rate"] == 0.0
    assert (table["wrong"] == 0).all()


def test_hard_phrasings_lower_recall_not_precision(sample):
    table, summary = evaluate_extraction(
        MockLLM(rule_based_responder), sample.texts, sample.gold
    )
    assert summary["field_accuracy"] < 0.95
    assert table["missed"].max() > 0.1
    assert table["wrong"].max() < 0.05
    assert summary["usage"]["calls"] == len(sample.texts)


def _mock(edit):
    def respond(system, user):
        out = rule_based_extract(application_from_prompt(user))
        edit(out["fields"], application_from_prompt(user))
        return json.dumps(out)

    return MockLLM(respond)


def test_unsupported_values_are_rejected_not_counted_wrong(sample):
    def off_by_one(fields, text):
        if fields["age"]["value"] is not None:
            fields["age"] = {"value": fields["age"]["value"] + 1, "evidence": text}

    table, _ = evaluate_extraction(_mock(off_by_one), sample.texts, sample.gold)
    row = table.set_index("field").loc["age"]
    assert row["wrong"] == 0
    assert row["rejected"] > 0


def test_misattributed_quotes_pass_validation_and_count_as_wrong(sample):
    # The quote check cannot tell which field a stated number belongs to.
    def swap(fields, text):
        if fields["open_credit"]["value"] is not None:
            fields["dependents"] = dict(fields["open_credit"])

    table, _ = evaluate_extraction(_mock(swap), sample.texts, sample.gold)
    assert table.set_index("field").loc["dependents", "wrong"] > 0.2


def test_cli_extract_eval(tmp_path, capsys):
    out = tmp_path / "extraction.csv"
    assert main(["extract-eval", "--n", "40", "--output", str(out)]) == 0
    assert out.exists()
    text = capsys.readouterr().out
    assert "field_accuracy=" in text and "monthly_inc" in text
