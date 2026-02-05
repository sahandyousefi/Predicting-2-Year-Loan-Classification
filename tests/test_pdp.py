import numpy as np

from credit_risk.data.schema import FEATURES
from credit_risk.explain.pdp import (
    centered_ice,
    ice_curves,
    interaction_strength,
    partial_dependence,
)
from credit_risk.features.engineering import (
    FEATURE_COLUMNS,
    add_engineered_features,
    build_xy,
)
from credit_risk.models.registry import build_model


def _transform(df):
    return add_engineered_features(df)[FEATURE_COLUMNS]


def test_pd_rises_with_late_payments(credit_df):
    X, y = build_xy(credit_df)
    model = build_model("logistic").fit(X, y)
    ice = ice_curves(
        model,
        credit_df[FEATURES],
        "late_90",
        grid=np.arange(0, 5),
        transform=_transform,
    )
    pdp = partial_dependence(ice).to_numpy()
    assert (np.diff(pdp) > 0).all()


def test_ice_shape_and_centering(credit_df):
    X, y = build_xy(credit_df)
    model = build_model("hist_gbm").fit(X, y)
    ice = ice_curves(model, credit_df[FEATURES], "age", transform=_transform, sample=50)
    assert ice.shape[0] == 50
    assert (centered_ice(ice).iloc[:, 0] == 0).all()
    assert interaction_strength(ice) >= 0
