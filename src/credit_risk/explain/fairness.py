from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AGE_BINS = [0, 25, 35, 45, 55, 62, np.inf]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-61", "62+"]


def age_band(age: pd.Series) -> pd.Series:
    """Age bands; 62 is the ECOA threshold for elderly applicants."""
    return pd.cut(age, AGE_BINS, labels=AGE_LABELS, right=False)


def group_report(
    y_true, pd_hat, groups: pd.Series, threshold: float, min_group: int = 30
) -> pd.DataFrame:
    """Approval and error rates by group for an approve-if-PD-below rule.

    ``good_approval_rate`` is the share of applicants who did not default and
    were approved, the equal opportunity quantity for a lending decision.
    Disparate impact compares each group's approval rate with the group that
    has the highest rate; the four-fifths rule flags ratios below 0.8.
    """
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true),
            "p": np.asarray(pd_hat),
            "group": np.asarray(groups),
        }
    )
    df["approved"] = df["p"] < threshold
    rows = []
    for g, part in df.groupby("group", observed=True):
        if len(part) < min_group:
            continue
        goods = part[part["y"] == 0]
        bads = part[part["y"] == 1]
        rows.append(
            {
                "group": g,
                "n": len(part),
                "default_rate": part["y"].mean(),
                "mean_pd": part["p"].mean(),
                "approval_rate": part["approved"].mean(),
                "good_approval_rate": goods["approved"].mean()
                if len(goods)
                else np.nan,
                "bad_approval_rate": bads["approved"].mean() if len(bads) else np.nan,
                "auc": roc_auc_score(part["y"], part["p"])
                if part["y"].nunique() == 2
                else np.nan,
            }
        )
    report = pd.DataFrame(rows)
    ref = report.loc[report["approval_rate"].idxmax()]
    report["disparate_impact"] = report["approval_rate"] / ref["approval_rate"]
    report["equal_opportunity_diff"] = (
        report["good_approval_rate"] - ref["good_approval_rate"]
    )
    report["four_fifths_flag"] = report["disparate_impact"] < 0.8
    return report.round(4)
