from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from credit_risk.config import CostConfig
from credit_risk.evaluation.bootstrap import (
    DEFAULT_METRICS,
    bootstrap_indices,
    decision_metrics,
    metric_intervals,
)
from credit_risk.evaluation.calibration import calibration_summary
from credit_risk.evaluation.delong import delong_test
from credit_risk.evaluation.metrics import credit_metrics
from credit_risk.evaluation.thresholds import ThresholdResult, optimal_threshold
from credit_risk.models.registry import available_models, build_model

GBM_MODELS = {"hist_gbm", "xgboost", "lightgbm"}


def _score_models(
    names: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cost: CostConfig,
    monotone: bool,
) -> dict[str, tuple[np.ndarray, ThresholdResult]]:
    scored = {}
    for name in names:
        if name not in available_models():
            continue
        columns = list(X_train.columns) if monotone and name in GBM_MODELS else None
        model = build_model(name, monotone_columns=columns).fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        scored[name] = (p, optimal_threshold(y_test, p, cost))
    return scored


def _table(
    scored: dict[str, tuple[np.ndarray, ThresholdResult]], y_test: pd.Series
) -> pd.DataFrame:
    rows = []
    for name, (p, thr) in scored.items():
        rows.append(
            {
                "model": name,
                **credit_metrics(y_test, p),
                **{
                    k: v
                    for k, v in calibration_summary(y_test, p).items()
                    if k in {"brier", "ece"}
                },
                "threshold": round(thr.threshold, 3),
                "approval_rate": round(thr.approval_rate, 3),
                "profit_per_applicant": round(thr.profit_per_applicant, 4),
            }
        )
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best = table["model"].iloc[0]
    table["delong_p_vs_best"] = [
        float("nan")
        if m == best
        else round(delong_test(y_test, scored[best][0], scored[m][0]).p_value, 4)
        for m in table["model"]
    ]
    return table.reset_index(drop=True)


def compare_models(
    names: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cost: CostConfig | None = None,
    monotone: bool = False,
) -> pd.DataFrame:
    """Fit each model on the same split and report discrimination,
    calibration, a DeLong test against the best AUC, and realized profit at the
    profit-maximizing threshold.

    With ``monotone=True`` the gradient boosting models are fitted with the
    expected-direction constraints; the linear models are unchanged.
    """
    cost = cost or CostConfig()
    scored = _score_models(names, X_train, y_train, X_test, y_test, cost, monotone)
    return _table(scored, y_test)


def compare_models_with_intervals(
    names: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cost: CostConfig | None = None,
    monotone: bool = False,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Comparison table plus bootstrap intervals for every reported metric.

    All models are evaluated on the same stratified bootstrap samples of the
    test set. Decision metrics use each model's cut-off from the full test
    set, held fixed across replicates.
    """
    cost = cost or CostConfig()
    scored = _score_models(names, X_train, y_train, X_test, y_test, cost, monotone)
    y = np.asarray(y_test, dtype=int)
    indices = bootstrap_indices(y, n_boot, stratified=True, seed=seed)
    frames = []
    for name, (p, thr) in scored.items():
        metrics = {**DEFAULT_METRICS, **decision_metrics(thr.threshold, cost)}
        ci = metric_intervals(y, p, metrics, alpha=alpha, indices=indices)
        frames.append(ci.assign(model=name))
    intervals = pd.concat(frames, ignore_index=True)
    columns = ["model", "metric", "estimate", "se", "ci_low", "ci_high"]
    return _table(scored, y_test), intervals[columns]


def to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, sep, *body])
