import numpy as np
import pytest
from scipy.special import logit

from credit_risk.explain.shap_explain import explain, global_importance
from credit_risk.features.engineering import build_xy
from credit_risk.models.registry import available_models, build_model


@pytest.fixture(scope="module")
def xy(credit_df):
    return build_xy(credit_df)


@pytest.mark.parametrize("name", ["logistic", "scorecard", "hist_gbm", "xgboost"])
def test_attributions_are_additive(name, xy):
    if name in {"hist_gbm", "xgboost"}:
        pytest.importorskip("shap")
    if name not in available_models():
        pytest.skip(f"{name} not installed")
    X, y = xy
    model = build_model(name).fit(X, y)
    attr = explain(model, X.iloc[:50], background=X)
    expected = logit(np.clip(model.predict_proba(X.iloc[:50])[:, 1], 1e-12, 1 - 1e-12))
    np.testing.assert_allclose(attr.log_odds().to_numpy(), expected, atol=1e-4)


def test_global_importance_finds_late_payments(xy):
    X, y = xy
    model = build_model("logistic").fit(X, y)
    top = global_importance(explain(model, X)).index[:5]
    assert "late_severity" in top
