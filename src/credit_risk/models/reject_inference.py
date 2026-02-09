"""Reject inference for application scorecards.

A model fitted on approved applicants only (known good/bad, KGB) learns the
default process of the approved population. Reject inference adds the
rejected applicants back with inferred outcomes or reweights the approved
ones. Every method rests on an untestable assumption about the rejects:

- reweighting is consistent when selection depends only on the model inputs;
- fuzzy and hard-cutoff augmentation extrapolate the KGB model to the rejects
  and add no information about them, so they mainly change the intercept and
  the weight on regions with few approvals;
- parceling scales the KGB bad rate in each score band by a factor that is a
  business judgment.

None of them can recover information that drove the historical decision and
is absent from the data (Hand and Henley, 1993). The synthetic study in
:func:`reject_inference_study` makes this visible because the outcomes of the
rejects are known.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data.schema import TARGET
from credit_risk.data.through_the_door import ACCEPTED
from credit_risk.features.engineering import FEATURE_COLUMNS, add_engineered_features
from credit_risk.models.linear import make_logistic

METHODS = ("kgb", "reweighting", "fuzzy", "hard_cutoff", "parceling")


def _logistic() -> Pipeline:
    # Unweighted classes so PDs stay on the population scale.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", make_logistic(C=1.0, l1_ratio=0.0, class_weight=None)),
        ]
    )


def _fit(X: pd.DataFrame, y: np.ndarray, w: np.ndarray | None = None) -> Pipeline:
    return _logistic().fit(X, y, clf__sample_weight=w)


@dataclass
class AugmentedSample:
    X: pd.DataFrame
    y: np.ndarray
    weight: np.ndarray


def fuzzy_augmentation(
    X_acc: pd.DataFrame, y_acc: np.ndarray, X_rej: pd.DataFrame, kgb: Pipeline
) -> AugmentedSample:
    """Each reject enters twice, as bad with weight ``p`` and good with ``1 - p``."""
    p = kgb.predict_proba(X_rej)[:, 1]
    X = pd.concat([X_acc, X_rej, X_rej], ignore_index=True)
    y = np.concatenate([y_acc, np.ones(len(X_rej)), np.zeros(len(X_rej))])
    w = np.concatenate([np.ones(len(X_acc)), p, 1 - p])
    return AugmentedSample(X, y.astype(int), w)


def hard_cutoff_augmentation(
    X_acc: pd.DataFrame,
    y_acc: np.ndarray,
    X_rej: pd.DataFrame,
    kgb: Pipeline,
    cutoff: float,
) -> AugmentedSample:
    """Rejects with KGB PD above ``cutoff`` are labeled bad, the rest good."""
    p = kgb.predict_proba(X_rej)[:, 1]
    X = pd.concat([X_acc, X_rej], ignore_index=True)
    y = np.concatenate([y_acc, (p > cutoff).astype(int)])
    return AugmentedSample(X, y, np.ones(len(X)))


def parceling(
    X_acc: pd.DataFrame,
    y_acc: np.ndarray,
    X_rej: pd.DataFrame,
    kgb: Pipeline,
    inflation: float = 1.5,
    n_bands: int = 10,
    seed: int = 0,
) -> AugmentedSample:
    """Assign reject outcomes at ``inflation`` times the KGB band bad rate.

    Rejects are grouped into quantile bands of their KGB PD. Within a band,
    each reject is labeled bad with probability equal to the mean KGB PD of
    the band multiplied by ``inflation`` and capped at one. An inflation
    factor above one encodes the belief that rejects are riskier than
    approved applicants with the same score.
    """
    if inflation < 1:
        raise ValueError("inflation must be at least 1")
    rng = np.random.default_rng(seed)
    p_rej = kgb.predict_proba(X_rej)[:, 1]
    edges = np.quantile(p_rej, np.linspace(0, 1, n_bands + 1))
    band = np.clip(np.searchsorted(edges, p_rej, side="right") - 1, 0, n_bands - 1)
    count = np.bincount(band, minlength=n_bands)
    band_rate = np.bincount(band, p_rej, n_bands) / np.maximum(count, 1)
    p_bad = np.minimum(inflation * band_rate[band], 1.0)
    y_rej = (rng.uniform(size=len(X_rej)) < p_bad).astype(int)
    X = pd.concat([X_acc, X_rej], ignore_index=True)
    y = np.concatenate([y_acc, y_rej])
    return AugmentedSample(X, y, np.ones(len(X)))


def acceptance_weights(
    X_all: pd.DataFrame, accepted: np.ndarray, clip: float = 20.0
) -> np.ndarray:
    """Inverse probability of approval for the approved applicants.

    The approval model is a logistic regression on the same inputs. Weights
    are capped at ``clip`` to limit the variance from applicants the old
    policy almost never approved.
    """
    model = _fit(X_all, accepted.astype(int))
    p_acc = model.predict_proba(X_all[accepted])[:, 1]
    return np.minimum(1.0 / np.maximum(p_acc, 1e-6), clip)


def fit_reject_inference(
    method: str,
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    accepted: np.ndarray,
    inflation: float = 1.5,
    seed: int = 0,
) -> Pipeline:
    """Fit a logistic model with the given reject inference method.

    ``y_all`` is read only for approved applicants. Use ``"oracle"`` to fit on
    every outcome, which is only possible with synthetic data and gives the
    upper bound.
    """
    accepted = np.asarray(accepted, dtype=bool)
    y_all = np.asarray(y_all, dtype=int)
    if method == "oracle":
        return _fit(X_all, y_all)
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}', choose from {METHODS}")
    X_acc, y_acc = X_all[accepted], y_all[accepted]
    X_rej = X_all[~accepted]
    if method == "reweighting":
        return _fit(X_acc, y_acc, acceptance_weights(X_all, accepted))
    kgb = _fit(X_acc, y_acc)
    if method == "kgb":
        return kgb
    if method == "fuzzy":
        sample = fuzzy_augmentation(X_acc, y_acc, X_rej, kgb)
    elif method == "hard_cutoff":
        cutoff = float(np.quantile(kgb.predict_proba(X_acc)[:, 1], 0.9))
        sample = hard_cutoff_augmentation(X_acc, y_acc, X_rej, kgb, cutoff)
    else:
        sample = parceling(X_acc, y_acc, X_rej, kgb, inflation, seed=seed)
    return _fit(sample.X, sample.y, sample.weight)


def reject_inference_study(
    df: pd.DataFrame,
    methods: Sequence[str] = ("oracle", *METHODS),
    inflation: float = 1.5,
    test_size: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare methods on a held-out through-the-door sample.

    The test sample contains approved and rejected applicants with their true
    outcomes. The table reports AUC and Brier score on all applicants, AUC on
    the rejects alone, and the mean PD against the observed bad rate of the
    rejects, which shows whether a method corrects the optimism of KGB.
    """
    fe = add_engineered_features(df)
    X = fe[FEATURE_COLUMNS]
    y = fe[TARGET].to_numpy(dtype=int)
    acc = fe[ACCEPTED].to_numpy(dtype=bool)
    strata = 2 * acc + y
    idx_tr, idx_te = train_test_split(
        np.arange(len(fe)), test_size=test_size, stratify=strata, random_state=seed
    )
    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    y_te, rej_te = y[idx_te], ~acc[idx_te]

    rows = []
    for method in methods:
        model = fit_reject_inference(
            method, X_tr, y[idx_tr], acc[idx_tr], inflation, seed
        )
        p = model.predict_proba(X_te)[:, 1]
        rows.append(
            {
                "method": method,
                "auc_all": float(roc_auc_score(y_te, p)),
                "brier_all": float(brier_score_loss(y_te, p)),
                "auc_rejects": float(roc_auc_score(y_te[rej_te], p[rej_te])),
                "mean_pd_rejects": float(p[rej_te].mean()),
                "bad_rate_rejects": float(y_te[rej_te].mean()),
                "mean_pd_all": float(p.mean()),
                "bad_rate_all": float(y_te.mean()),
            }
        )
    return pd.DataFrame(rows)
