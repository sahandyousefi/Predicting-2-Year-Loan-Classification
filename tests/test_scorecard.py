import numpy as np
from sklearn.metrics import roc_auc_score

from credit_risk.data.schema import FEATURES, TARGET
from credit_risk.models.scorecard import Scorecard


def test_scorecard_ranks_risk(credit_df):
    X, y = credit_df[FEATURES], credit_df[TARGET]
    card = Scorecard().fit(X, y)
    # Higher score means lower risk, so AUC on the negated score.
    assert roc_auc_score(y, -card.score(X)) > 0.7


def test_points_reproduce_model_log_odds(credit_df):
    X, y = credit_df[FEATURES], credit_df[TARGET]
    card = Scorecard().fit(X, y)
    pd_from_points = card.score_to_pd(card.score(X))
    pd_model = card.predict_proba(X)[:, 1]
    # Points are rounded per bin, so allow a small tolerance.
    assert np.corrcoef(pd_from_points, pd_model)[0, 1] > 0.99


def test_pdo_scaling():
    card = Scorecard(base_score=600, base_odds=50, pdo=20)
    assert np.isclose(card.score_to_pd(600), 1 / 51)
    assert np.isclose(card.score_to_pd(620), 1 / 101)


def test_points_table_covers_all_features(credit_df):
    card = Scorecard().fit(credit_df[FEATURES], credit_df[TARGET])
    assert set(card.points_table_["feature"]) == set(FEATURES)
