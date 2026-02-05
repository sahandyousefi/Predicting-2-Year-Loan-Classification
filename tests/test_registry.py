import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk.features.engineering import build_xy
from credit_risk.models.registry import available_models, build_model


def test_core_models_always_available():
    assert {"logistic", "scorecard", "hist_gbm"} <= set(available_models())


def test_unknown_model():
    with pytest.raises(KeyError):
        build_model("svm")


@pytest.mark.parametrize(
    "name", ["logistic", "scorecard", "hist_gbm", "xgboost", "lightgbm"]
)
def test_models_fit_and_rank(name, credit_df):
    if name not in available_models():
        pytest.skip(f"{name} not installed")
    X, y = build_xy(credit_df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    model = build_model(name).fit(X_tr, y_tr)
    assert roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]) > 0.65
