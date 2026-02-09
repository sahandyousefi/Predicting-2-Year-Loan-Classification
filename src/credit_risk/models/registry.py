from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier

from credit_risk.models.monotonic import constraint_vector, model_kwargs
from credit_risk.models.scorecard import Scorecard
from credit_risk.models.train import build_pipeline

SEED = 42


def _logistic(**params: Any):
    return build_pipeline(**params)


def _scorecard(**params: Any):
    return Scorecard(**params)


def _hist_gbm(**params: Any):
    defaults = dict(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0
    )
    return HistGradientBoostingClassifier(random_state=SEED, **{**defaults, **params})


def _xgboost(**params: Any):
    from xgboost import XGBClassifier

    defaults = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric="auc",
        n_jobs=1,
    )
    return XGBClassifier(random_state=SEED, **{**defaults, **params})


def _lightgbm(**params: Any):
    from lightgbm import LGBMClassifier

    defaults = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=30,
        n_jobs=1,
        verbose=-1,
    )
    return LGBMClassifier(random_state=SEED, **{**defaults, **params})


_BUILDERS: dict[str, tuple[Callable[..., Any], str | None]] = {
    "logistic": (_logistic, None),
    "scorecard": (_scorecard, None),
    "hist_gbm": (_hist_gbm, None),
    "xgboost": (_xgboost, "xgboost"),
    "lightgbm": (_lightgbm, "lightgbm"),
}


def available_models() -> list[str]:
    return [
        name
        for name, (_, module) in _BUILDERS.items()
        if module is None or importlib.util.find_spec(module) is not None
    ]


def build_model(
    name: str, monotone_columns: Sequence[str] | None = None, **params: Any
):
    """Return an unfitted estimator by name.

    XGBoost and LightGBM are optional: install them with ``pip install -e
    ".[gbm]"``.

    Parameters
    ----------
    name
        Registry key.
    monotone_columns
        Column order of the training frame. When given, the gradient boosting
        models are fitted with the expected-direction constraints from
        :mod:`credit_risk.models.monotonic`.
    **params
        Passed to the estimator and override the defaults.
    """
    if name not in _BUILDERS:
        raise KeyError(f"unknown model '{name}', choose from {sorted(_BUILDERS)}")
    builder, module = _BUILDERS[name]
    if module and importlib.util.find_spec(module) is None:
        raise ImportError(f"'{name}' needs the {module} package: pip install -e .[gbm]")
    if monotone_columns is not None:
        params = {**model_kwargs(name, constraint_vector(monotone_columns)), **params}
    return builder(**params)
