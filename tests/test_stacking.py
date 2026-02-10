import joblib
import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk.features.engineering import build_xy
from credit_risk.models.registry import build_model
from credit_risk.models.stacking import StackedEnsemble, fit_nonnegative_logistic


@pytest.fixture(scope="module")
def split(credit_df):
    X, y = build_xy(credit_df)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


@pytest.fixture(scope="module")
def stack(split):
    X_tr, _, y_tr, _ = split
    return StackedEnsemble(("logistic", "hist_gbm"), n_splits=3).fit(X_tr, y_tr)


def test_nonnegative_logistic_recovers_positive_weights():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(5000, 2))
    y = (rng.uniform(size=5000) < expit(-0.5 + 1.5 * Z[:, 0])).astype(int)
    b0, w = fit_nonnegative_logistic(Z, y)
    assert b0 == pytest.approx(-0.5, abs=0.15)
    assert w[0] == pytest.approx(1.5, abs=0.15)
    assert w[1] == pytest.approx(0.0, abs=0.1)


def test_negative_signal_is_clipped_to_zero():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(3000, 1))
    y = (rng.uniform(size=3000) < expit(-Z[:, 0])).astype(int)
    _, w = fit_nonnegative_logistic(Z, y)
    assert w[0] == pytest.approx(0.0, abs=1e-8)


def test_oof_predictions_and_weights(stack, split):
    X_tr, _, y_tr, _ = split
    assert stack.oof_.shape == (len(X_tr), 2)
    assert (stack.coef_ >= 0).all()
    assert set(stack.weights) == {"logistic", "hist_gbm"}
    assert set(stack.oof_auc_) == {"logistic", "hist_gbm", "stacked"}


def test_stack_ranks_and_calibrates(stack, split):
    _, X_te, _, y_te = split
    p = stack.predict_proba(X_te)[:, 1]
    base = stack.base_predictions(X_te)
    assert (
        roc_auc_score(y_te, p)
        > max(roc_auc_score(y_te, base[:, j]) for j in range(2)) - 0.01
    )
    # The balanced-weight logistic base model is miscalibrated; the stack is not.
    assert brier_score_loss(y_te, p) < brier_score_loss(y_te, base[:, 0])
    assert p.mean() == pytest.approx(np.mean(y_te), abs=0.05)


def test_registry_and_persistence(split, tmp_path):
    X_tr, X_te, y_tr, _ = split
    model = build_model("stacked", base_models=("logistic", "scorecard"), n_splits=3)
    model.fit(X_tr, y_tr)
    joblib.dump(model, tmp_path / "stack.joblib")
    restored = joblib.load(tmp_path / "stack.joblib")
    assert np.allclose(restored.predict_proba(X_te), model.predict_proba(X_te))


@pytest.mark.parametrize("bases", [("logistic",), ("logistic", "stacked")])
def test_invalid_base_models(split, bases):
    X_tr, _, y_tr, _ = split
    with pytest.raises(ValueError):
        StackedEnsemble(bases).fit(X_tr, y_tr)
