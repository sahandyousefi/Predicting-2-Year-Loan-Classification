import numpy as np
import pandas as pd

TARGET = "dlq_2yrs"

BASE_FEATURES = [
    "age",
    "debt_ratio",
    "monthly_inc",
    "open_credit",
    "rev_util",
    "dependents",
]

ENGINEERED_FEATURES = [
    "debt_per_dependent",
    "inc_per_dependent",
    "credit_to_income",
    "late_severity",
    "log_monthly_inc",
    "log_rev_util",
    "log_debt_ratio",
]

FEATURE_COLUMNS = BASE_FEATURES + ENGINEERED_FEATURES

REQUIRED_COLUMNS = BASE_FEATURES + ["late_30_59", "late_60_89", "late_90"]

LATE_WEIGHTS = {"late_30_59": 1, "late_60_89": 2, "late_90": 3}


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)
    out = df.copy()
    household = out["dependents"].clip(lower=0) + 1

    out["debt_per_dependent"] = out["debt_ratio"] / household
    out["inc_per_dependent"] = out["monthly_inc"] / household
    out["credit_to_income"] = out["open_credit"] / (out["monthly_inc"].clip(lower=0) + 1)
    out["late_severity"] = sum(out[col] * w for col, w in LATE_WEIGHTS.items())

    out["log_monthly_inc"] = np.log1p(out["monthly_inc"].clip(lower=0))
    out["log_rev_util"] = np.log1p(out["rev_util"].clip(lower=0))
    out["log_debt_ratio"] = np.log1p(out["debt_ratio"].clip(lower=0))
    return out


def build_xy(df: pd.DataFrame):
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")
    fe = add_engineered_features(df)
    X = fe[FEATURE_COLUMNS].copy()
    y = fe[TARGET].astype(int).copy()
    return X, y
