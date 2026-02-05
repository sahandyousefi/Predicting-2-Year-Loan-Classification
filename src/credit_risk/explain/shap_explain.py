from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.models.scorecard import Scorecard


@dataclass
class Attribution:
    """Additive attributions in log-odds of default.

    For every row, ``base_value + values.sum(axis=1)`` equals the model's
    log-odds prediction.
    """

    values: pd.DataFrame
    base_value: float

    def log_odds(self) -> pd.Series:
        return self.base_value + self.values.sum(axis=1)


def _linear(coef: np.ndarray, intercept: float, Z, Z_bg) -> tuple[np.ndarray, float]:
    # With independent features, exact SHAP values of a linear model are
    # coef * (x - E[x]) (Lundberg and Lee, 2017).
    mean_bg = np.asarray(Z_bg, dtype=float).mean(axis=0)
    values = coef * (np.asarray(Z, dtype=float) - mean_bg)
    return values, float(intercept + coef @ mean_bg)


def explain(
    model, X: pd.DataFrame, background: pd.DataFrame | None = None
) -> Attribution:
    """Local attributions for logistic pipelines, scorecards and tree ensembles.

    Linear models are explained in closed form against ``background`` (the
    training sample). Tree ensembles use TreeSHAP and need the ``explain``
    extra.
    """
    background = X if background is None else background

    if isinstance(model, Pipeline) and hasattr(model[-1], "coef_"):
        transform = model[:-1]
        clf = model[-1]
        values, base = _linear(
            clf.coef_[0],
            clf.intercept_[0],
            transform.transform(X),
            transform.transform(background),
        )
    elif isinstance(model, Scorecard):
        clf = model.model_
        values, base = _linear(
            clf.coef_[0],
            clf.intercept_[0],
            model.binner_.transform(X),
            model.binner_.transform(background),
        )
    else:
        import shap

        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(X))
        if values.ndim == 3:  # some versions return one array per class
            values = values[..., -1]
        base = float(np.ravel(explainer.expected_value)[-1])

    return Attribution(pd.DataFrame(values, columns=X.columns, index=X.index), base)


def global_importance(attr: Attribution) -> pd.Series:
    """Mean absolute attribution per feature, largest first."""
    return attr.values.abs().mean().sort_values(ascending=False)
