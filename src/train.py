import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

from src.features import FEATURE_COLUMNS, build_xy

RANDOM_STATE = 42


def build_pipeline(C: float = 100.0, penalty: str = "l1") -> Pipeline:
    # Scaling lives inside the pipeline so it is fit on training folds only
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                ),
            ),
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


def run(data_path: str, output_dir: str = "artifacts", test_size: float = 0.2) -> dict:
    df = pd.read_csv(data_path)
    X, y = build_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
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

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Train the 2-year delinquency classifier")
    parser.add_argument("--data", required=True, help="Path to Credit Risk Benchmark CSV")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.data, args.output_dir, args.test_size)
    print(json.dumps(result, indent=2))
