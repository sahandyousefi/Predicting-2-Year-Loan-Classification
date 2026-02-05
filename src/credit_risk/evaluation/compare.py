from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from credit_risk.config import CostConfig
from credit_risk.evaluation.calibration import calibration_summary
from credit_risk.evaluation.delong import delong_test
from credit_risk.evaluation.metrics import credit_metrics
from credit_risk.evaluation.thresholds import optimal_threshold
from credit_risk.models.registry import available_models, build_model


def compare_models(
    names: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cost: CostConfig | None = None,
) -> pd.DataFrame:
    """Fit each model on the same split and report discrimination,
    calibration, a DeLong test against the best AUC, and realized profit at the
    profit-maximizing threshold."""
    cost = cost or CostConfig()
    probs: dict[str, pd.Series] = {}
    rows = []
    for name in names:
        if name not in available_models():
            continue
        model = build_model(name).fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        probs[name] = p
        thr = optimal_threshold(y_test, p, cost)
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
        else round(delong_test(y_test, probs[best], probs[m]).p_value, 4)
        for m in table["model"]
    ]
    return table.reset_index(drop=True)


def to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, sep, *body])
