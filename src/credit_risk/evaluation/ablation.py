"""Ablation of text embedding features against the structured baseline.

Three feature sets are compared under the same repeated stratified K-fold
splits: structured attributes only, text embeddings only, and both. The
difference in fold ROC-AUC against the structured baseline is tested with the
corrected resampled t-test of Nadeau and Bengio (2003), which inflates the
variance for the overlap between training sets across folds.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data.schema import TARGET
from credit_risk.data.synthetic_text import TEXT_COLUMN, add_purpose_text
from credit_risk.features.engineering import FEATURE_COLUMNS, add_engineered_features
from credit_risk.features.text_embedding import TextEmbedder
from credit_risk.models.linear import make_logistic

FEATURE_SETS = ("structured", "text", "combined")
LEARNERS = ("logistic", "hist_gbm")


def build_text_pipeline(
    feature_set: str,
    learner: str = "logistic",
    n_components: int = 16,
    backend: str = "tfidf",
    seed: int = 0,
) -> Pipeline:
    """Pipeline over a frame holding engineered features and the text column."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    if learner not in LEARNERS:
        raise ValueError(f"learner must be one of {LEARNERS}")
    parts = []
    if feature_set in ("structured", "combined"):
        parts.append(("num", StandardScaler(), list(FEATURE_COLUMNS)))
    if feature_set in ("text", "combined"):
        embedder = TextEmbedder(
            backend=backend, n_components=n_components, random_state=seed
        )
        parts.append(("txt", embedder, TEXT_COLUMN))
    clf = (
        make_logistic(C=1.0, l1_ratio=0.0, random_state=seed)
        if learner == "logistic"
        else HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15, random_state=seed
        )
    )
    return Pipeline([("features", ColumnTransformer(parts)), ("clf", clf)])


def corrected_resampled_ttest(
    diffs: np.ndarray, n_train: int, n_test: int
) -> tuple[float, float]:
    """Nadeau-Bengio corrected t statistic and two-sided p-value.

    ``diffs`` are per-fold score differences from ``J`` resamples. The
    variance factor ``1/J + n_test/n_train`` replaces the naive ``1/J``.
    """
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt((1 / j + n_test / n_train) * var)
    return float(stat), float(2 * student_t.sf(abs(stat), df=j - 1))


def text_ablation(
    df: pd.DataFrame,
    learners: Sequence[str] = ("logistic",),
    n_components: Sequence[int] = (16,),
    backend: str = "tfidf",
    n_splits: int = 5,
    n_repeats: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Cross-validated AUC of each feature set and its gain over structured.

    ``df`` must contain the raw attributes, the target and ``purpose_text``.
    """
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"'{TEXT_COLUMN}' column is required")
    X = add_engineered_features(df)[list(FEATURE_COLUMNS) + [TEXT_COLUMN]]
    y = df[TARGET].astype(int).to_numpy()
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    splits = list(cv.split(X, y))
    n_test = int(np.mean([len(te) for _, te in splits]))
    n_train = len(y) - n_test

    configs = [("structured", 0)] + [
        (fs, k) for fs in ("text", "combined") for k in n_components
    ]
    rows = []
    for learner in learners:
        fold_auc: dict[tuple[str, int], np.ndarray] = {}
        for fs, k in configs:
            scores = []
            for tr, te in splits:
                pipe = build_text_pipeline(fs, learner, k or 1, backend, seed)
                pipe.fit(X.iloc[tr], y[tr])
                p = pipe.predict_proba(X.iloc[te])[:, 1]
                scores.append(roc_auc_score(y[te], p))
            fold_auc[(fs, k)] = np.asarray(scores)
        base = fold_auc[("structured", 0)]
        for (fs, k), auc in fold_auc.items():
            stat, p_value = (
                (float("nan"), float("nan"))
                if fs == "structured"
                else corrected_resampled_ttest(auc - base, n_train, n_test)
            )
            rows.append(
                {
                    "learner": learner,
                    "feature_set": fs,
                    "text_dim": k,
                    "auc_mean": float(auc.mean()),
                    "auc_std": float(auc.std(ddof=1)),
                    "gain_vs_structured": float((auc - base).mean()),
                    "t_stat": stat,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def signal_sweep(
    df: pd.DataFrame,
    signals: Sequence[float] = (0.0, 0.5, 1.0),
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Repeat the ablation with synthetic text of increasing signal strength.

    ``signal = 0`` is a negative control: the text is independent of the
    outcome, so the combined model should show no gain.
    """
    frames = []
    for s in signals:
        table = text_ablation(add_purpose_text(df, s, seed), seed=seed, **kwargs)
        frames.append(table.assign(signal=s))
    out = pd.concat(frames, ignore_index=True)
    return out[["signal", *[c for c in out.columns if c != "signal"]]]
