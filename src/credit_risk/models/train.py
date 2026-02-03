from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.config import Config, load_config
from credit_risk.data.io import load_dataset
from credit_risk.features.engineering import FEATURE_COLUMNS, build_xy
from credit_risk.models.linear import make_logistic


def build_pipeline(C: float = 100.0, l1_ratio: float = 1.0) -> Pipeline:
    # Scaling lives inside the pipeline so it is fit on training folds only
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", make_logistic(C=C, l1_ratio=l1_ratio)),
        ]
    )


def best_f1_threshold(y_true, y_prob) -> float:
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thresh[int(np.argmax(f1))])


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def run(
    data_path: str | None = None,
    output_dir: str | None = None,
    test_size: float | None = None,
    config: Config | None = None,
) -> dict:
    cfg = config or load_config()
    seed = cfg.data.random_state
    df = load_dataset(data_path or cfg.data.path)
    X, y = build_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size or cfg.data.test_size,
        stratify=y,
        random_state=seed,
    )

    pipe = build_pipeline(C=cfg.model.C, l1_ratio=cfg.model.l1_ratio)
    cv = StratifiedKFold(n_splits=cfg.model.cv_folds, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    pipe.fit(X_train, y_train)
    train_prob = pipe.predict_proba(X_train)[:, 1]
    test_prob = pipe.predict_proba(X_test)[:, 1]

    # Threshold is chosen on training data to avoid tuning on the test set
    tuned = best_f1_threshold(y_train, train_prob)

    coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=FEATURE_COLUMNS)
    report = {
        "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_roc_auc_std": round(float(cv_auc.std()), 4),
        "test_default_threshold": evaluate(y_test, test_prob, 0.5),
        "test_tuned_threshold": evaluate(y_test, test_prob, tuned),
        "coefficients": coefs.sort_values(ascending=False).round(4).to_dict(),
    }

    out = Path(output_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the 2-year delinquency classifier"
    )
    parser.add_argument("--config", default=None, help="Path to a YAML config")
    parser.add_argument("--data", default=None, help="Overrides data.path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test-size", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    result = run(args.data, args.output_dir, args.test_size, config=cfg)
    print(json.dumps(result, indent=2))
