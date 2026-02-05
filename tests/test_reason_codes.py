import pandas as pd

from credit_risk.explain.reason_codes import (
    FEATURE_GROUP,
    REASON_TEXT,
    group_attributions,
    reason_codes,
)
from credit_risk.explain.shap_explain import Attribution, explain
from credit_risk.features.engineering import FEATURE_COLUMNS, build_xy
from credit_risk.models.registry import build_model


def test_every_feature_has_a_reason_group():
    assert set(FEATURE_COLUMNS) <= set(FEATURE_GROUP)
    assert set(FEATURE_GROUP.values()) <= set(REASON_TEXT)


def test_grouping_preserves_total():
    values = pd.DataFrame({"late_90": [0.5], "late_severity": [0.2], "age": [-0.1]})
    grouped = group_attributions(Attribution(values, 0.0))
    assert grouped.loc[0, "DELINQ"] == 0.7
    assert grouped.sum(axis=1).iloc[0] == values.sum(axis=1).iloc[0]


def test_reasons_are_positive_and_ordered(credit_df):
    X, y = build_xy(credit_df)
    model = build_model("logistic").fit(X, y)
    attr = explain(model, X.iloc[:100], background=X)
    for reasons in reason_codes(attr, max_reasons=4):
        assert len(reasons) <= 4
        contribs = [r.contribution for r in reasons]
        assert all(c > 0 for c in contribs)
        assert contribs == sorted(contribs, reverse=True)


def test_high_risk_applicant_cites_delinquency(credit_df):
    X, y = build_xy(credit_df)
    model = build_model("logistic").fit(X, y)
    risky = X.iloc[[0]].copy()
    risky["late_severity"] = X["late_severity"].max()
    codes = [r.code for r in reason_codes(explain(model, risky, X))[0]]
    assert codes[0] == "DELINQ"
