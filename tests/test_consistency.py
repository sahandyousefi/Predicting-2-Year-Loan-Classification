import pytest

from credit_risk.cli import main
from credit_risk.data.synthetic_applications import make_applications
from credit_risk.llm.client import MockLLM
from credit_risk.llm.consistency import (
    CHECKS,
    bureau_reference,
    compare_field,
    misstatement_study,
    reconcile,
)
from credit_risk.llm.extraction import extract_fields
from credit_risk.llm.extraction_baseline import rule_based_responder

BUREAU = {
    "age": 41,
    "monthly_inc": 5000.0,
    "debt_ratio": 0.2,
    "rev_util": 0.6,
    "late_30_59": 2,
    "late_60_89": 0,
    "late_90": 1,
    "open_credit": 7,
    "real_estate": 1,
    "dependents": 2,
}
CHECK = {c.name: c for c in CHECKS}


def test_bureau_reference_derives_monthly_debt():
    ref = bureau_reference(BUREAU)
    assert ref["monthly_debt"] == pytest.approx(1000.0)
    assert bureau_reference({"debt_ratio": 0.2})["monthly_debt"] is None


@pytest.mark.parametrize(
    "name, stated, status, material",
    [
        ("monthly_inc", 5300.0, "consistent", False),
        ("monthly_inc", 7000.0, "overstated", True),
        ("monthly_inc", 3000.0, "understated", False),
        ("monthly_debt", 400.0, "understated", True),
        ("rev_util", 0.55, "consistent", False),
        ("rev_util", 0.2, "understated", True),
        ("late_90", 0.0, "understated", True),
        ("late_90", 3.0, "overstated", False),
        ("age", 43.0, "overstated", True),
        ("open_credit", 9.0, "overstated", False),
    ],
)
def test_direction_and_materiality(name, stated, status, material):
    ref = bureau_reference(BUREAU)[name]
    row = compare_field(CHECK[name], stated, ref)
    assert row["status"] == status
    assert row["material"] is material


def test_missing_values():
    assert compare_field(CHECK["age"], None, 41.0)["status"] == "not_stated"
    assert compare_field(CHECK["age"], 41.0, None)["status"] == "no_reference"


def test_reconcile_prefers_bureau_and_fills_gaps():
    stated = {"monthly_inc": 9000.0, "late_90": 0.0, "dependents": 2.0}
    bureau = {k: v for k, v in BUREAU.items() if k != "dependents"}
    report = reconcile(stated, bureau)
    assert report.needs_review
    assert report.material == ["INCOME_OVERSTATED", "DELINQUENCY_UNDERSTATED"]
    assert report.inputs["monthly_inc"] == 5000.0
    assert report.inputs["late_90"] == 1.0
    assert report.inputs["dependents"] == 2.0
    assert report.inputs["debt_ratio"] == 0.2
    frame = report.to_frame().set_index("field")
    assert frame.loc["dependents", "status"] == "no_reference"
    assert frame.loc["monthly_inc", "code"] == "INCOME_OVERSTATED"


def test_reconcile_without_bureau_ratio_uses_stated_debt():
    stated = {"monthly_inc": 5000.0, "monthly_debt": 1500.0}
    report = reconcile(stated, {"monthly_inc": 5000.0})
    assert report.inputs["debt_ratio"] == pytest.approx(0.3)
    assert not report.needs_review


def test_honest_applications_are_identical_for_any_rate():
    honest = make_applications(60, seed=4)
    mixed = make_applications(60, seed=4, misstatement_rate=0.5)
    assert all(m is None for m in honest.misstated)
    for i, kind in enumerate(mixed.misstated):
        if kind is None:
            assert mixed.texts[i] == honest.texts[i]
    assert any(kind is not None for kind in mixed.misstated)


def test_misstatements_are_detected_end_to_end():
    sample = make_applications(300, seed=1, hard_rate=0.0, misstatement_rate=0.4)
    llm = MockLLM(rule_based_responder)
    reports = [
        reconcile(extract_fields(text, llm).values, bureau)
        for text, (_, bureau) in zip(sample.texts, sample.attributes.iterrows())
    ]
    table = misstatement_study(reports, sample.misstated).set_index("misstatement")
    assert table.loc["honest", "flagged"] == 0.0
    # Detection is bounded by the share of applications that state the field.
    assert (table.drop(index="honest")["flagged"] > 0.6).all()


def test_cli_consistency(capsys):
    assert main(["consistency", "--n", "120"]) == 0
    assert "income_overstated" in capsys.readouterr().out
