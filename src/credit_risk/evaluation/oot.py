"""Out-of-time validation.

The model is developed on vintages before a cut-off month, with a random
in-time holdout, and evaluated on all later vintages. Comparing the in-time
and out-of-time results separates sampling error from the effect of
population change, which a random split cannot show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk.data.vintage import VINTAGE_COLUMN, out_of_time_split
from credit_risk.evaluation.delong import auc_confidence_interval
from credit_risk.evaluation.metrics import ks_statistic
from credit_risk.features.engineering import build_xy
from credit_risk.models.registry import build_model
from credit_risk.monitoring.drift import score_psi


@dataclass
class OOTReport:
    summary: pd.DataFrame
    by_vintage: pd.DataFrame
    score_stability: dict[str, float | str]


def _sample_metrics(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    auc, lo, hi = auc_confidence_interval(y, p)
    return {
        "sample": name,
        "n": len(y),
        "default_rate": float(y.mean()),
        "mean_pd": float(p.mean()),
        "roc_auc": auc,
        "auc_ci_low": lo,
        "auc_ci_high": hi,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
    }


def vintage_table(vintages: pd.Series, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Default rate, mean PD and AUC for each application month."""
    df = pd.DataFrame({"vintage": vintages.to_numpy(), "y": y, "p": p})
    rows = []
    for v, g in df.groupby("vintage", sort=True):
        rows.append(
            {
                "vintage": str(v),
                "n": len(g),
                "default_rate": float(g["y"].mean()),
                "mean_pd": float(g["p"].mean()),
                "roc_auc": float(roc_auc_score(g["y"], g["p"]))
                if g["y"].nunique() == 2
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def oot_validation(
    df: pd.DataFrame,
    cutoff: str,
    model: str = "logistic",
    holdout_size: float = 0.2,
    seed: int = 0,
) -> OOTReport:
    """Fit on vintages before ``cutoff`` and evaluate in time and out of time.

    Parameters
    ----------
    df : DataFrame
        Labeled data with a ``vintage`` column.
    cutoff : str
        First out-of-time month, ``YYYY-MM``.
    model : str
        Registry name of the model to fit.
    holdout_size : float
        Share of the development sample kept as the in-time holdout.
    seed : int
        Seed of the in-time split.
    """
    dev, oot = out_of_time_split(df, cutoff)
    X_dev, y_dev = build_xy(dev)
    X_oot, y_oot = build_xy(oot)
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X_dev, y_dev, test_size=holdout_size, stratify=y_dev, random_state=seed
    )
    fitted = build_model(model).fit(X_tr, y_tr)
    p_tr = fitted.predict_proba(X_tr)[:, 1]
    p_ho = fitted.predict_proba(X_ho)[:, 1]
    p_oot = fitted.predict_proba(X_oot)[:, 1]

    summary = pd.DataFrame(
        [
            _sample_metrics("in_time_holdout", y_ho.to_numpy(), p_ho),
            _sample_metrics("out_of_time", y_oot.to_numpy(), p_oot),
        ]
    )
    vintages = pd.concat([dev.loc[X_ho.index, VINTAGE_COLUMN], oot[VINTAGE_COLUMN]])
    by_vintage = vintage_table(
        vintages, np.concatenate([y_ho, y_oot]), np.concatenate([p_ho, p_oot])
    )
    first_oot = str(pd.Period(cutoff, freq="M"))
    by_vintage["period"] = np.where(
        by_vintage["vintage"] < first_oot, "in_time", "out_of_time"
    )
    return OOTReport(summary, by_vintage, score_psi(p_tr, p_oot))
