from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from credit_risk.models.registry import build_model


def suggest_params(trial, name: str) -> dict[str, Any]:
    """Search space per model family."""
    if name == "logistic":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "l1_ratio": trial.suggest_categorical("l1_ratio", [0.0, 1.0]),
        }
    if name == "hist_gbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 63, log=True),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 10, 200, log=True
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-3, 10, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
        }
    if name == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1, 50, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        }
    if name == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 63, log=True),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 200, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        }
    raise KeyError(f"no search space for '{name}'")


def tune(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 5,
    seed: int = 42,
) -> tuple[dict[str, Any], float]:
    """TPE search maximizing cross-validated ROC-AUC. Returns params and score."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)

    def objective(trial) -> float:
        model = build_model(name, **suggest_params(trial, name))
        return cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


@dataclass
class NestedCVResult:
    outer_scores: list[float] = field(default_factory=list)
    inner_scores: list[float] = field(default_factory=list)
    params: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.outer_scores))

    @property
    def std(self) -> float:
        return float(np.std(self.outer_scores, ddof=1))

    @property
    def optimism(self) -> float:
        """Average gap between inner (selection) and outer (test) AUC."""
        return float(np.mean(self.inner_scores) - self.mean)


def nested_cv(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_trials: int = 30,
    seed: int = 42,
) -> NestedCVResult:
    """Unbiased estimate of tuned-model performance.

    Hyperparameters are chosen inside each outer training fold only, so the
    outer test folds never influence model selection.
    """
    result = NestedCVResult()
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in outer.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        params, inner = tune(name, X_tr, y_tr, n_trials, inner_folds, seed)
        model = build_model(name, **params).fit(X_tr, y_tr)
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        result.outer_scores.append(float(roc_auc_score(y.iloc[test_idx], prob)))
        result.inner_scores.append(inner)
        result.params.append(params)
    return result
