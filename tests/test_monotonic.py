import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk.features.engineering import FEATURE_COLUMNS, build_xy
from credit_risk.models.monotonic import (
    EXPECTED_DIRECTION,
    constraint_vector,
    model_kwargs,
    monotonicity_violations,
    violation_table,
)
from credit_risk.models.registry import available_models, build_model


@pytest.fixture(scope="module")
def split(credit_df):
    X, y = build_xy(credit_df)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


def test_constraint_vector_follows_column_order():
    vec = constraint_vector(["late_severity", "age", "unknown"])
    assert vec == [1, -1, 0]
    assert len(constraint_vector(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


def test_overrides_are_validated():
    assert constraint_vector(["age"], {"age": 0}) == [0]
    with pytest.raises(ValueError):
        constraint_vector(["age"], {"age": 2})


def test_kwargs_per_library():
    assert model_kwargs("hist_gbm", [1, -1]) == {"monotonic_cst": [1, -1]}
    assert model_kwargs("xgboost", [1, 0, -1]) == {"monotone_constraints": "(1,0,-1)"}
    with pytest.raises(ValueError):
        model_kwargs("logistic", [1])


@pytest.mark.parametrize("name", ["hist_gbm", "xgboost", "lightgbm"])
def test_constrained_models_respect_directions(name, split):
    if name not in available_models():
        pytest.skip(f"{name} not installed")
    X_tr, X_te, y_tr, y_te = split
    model = build_model(name, monotone_columns=list(X_tr.columns)).fit(X_tr, y_tr)
    table = violation_table(model, X_te, n_rows=100)
    assert set(table["feature"]) == {
        c for c in X_tr.columns if EXPECTED_DIRECTION.get(c, 0) != 0
    }
    assert table["violation_share"].max() == 0.0
    assert roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]) > 0.65


def test_unconstrained_model_can_violate(split):
    X_tr, X_te, y_tr, _ = split
    model = build_model("hist_gbm", max_leaf_nodes=63, min_samples_leaf=5).fit(
        X_tr, y_tr
    )
    share = monotonicity_violations(model, X_te, "rev_util", 1, n_rows=100)
    assert 0.0 < share <= 1.0


def test_violation_direction_is_checked(split):
    X_tr, X_te, y_tr, _ = split
    model = build_model("hist_gbm").fit(X_tr, y_tr)
    with pytest.raises(ValueError):
        monotonicity_violations(model, X_te, "age", 0)
    assert np.isfinite(monotonicity_violations(model, X_te, "age", -1))
