from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov distance between score distributions of goods and bads.

    Equal to the maximum of TPR - FPR over all thresholds.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def gini(y_true, y_score) -> float:
    """Accuracy ratio, 2 * AUC - 1."""
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def lift_gain_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Rank-ordered bins (highest risk first) with gain and lift.

    ``cum_gain`` is the share of all events captured up to and including the
    bin; ``lift`` is the bin event rate divided by the overall event rate.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    df = df.sort_values("score", ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    df["bin"] = np.arange(len(df)) * n_bins // len(df) + 1
    base_rate = df["y"].mean()
    table = df.groupby("bin").agg(
        count=("y", "size"),
        events=("y", "sum"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    )
    table["event_rate"] = table["events"] / table["count"]
    table["cum_gain"] = table["events"].cumsum() / table["events"].sum()
    table["lift"] = table["event_rate"] / base_rate
    table["cum_lift"] = (table["events"].cumsum() / table["count"].cumsum()) / base_rate
    return table.reset_index()


def credit_metrics(y_true, y_score) -> dict[str, float]:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "gini": round(gini(y_true, y_score), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
    }
