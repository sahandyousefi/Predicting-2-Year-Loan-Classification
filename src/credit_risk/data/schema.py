from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET = "dlq_2yrs"

LATE_COLUMNS = ["late_30_59", "late_60_89", "late_90"]

# Hard bounds fail validation; soft bounds only raise a warning.
COLUMN_SPEC: dict[str, dict[str, tuple[float, float]]] = {
    "rev_util": {"hard": (0.0, np.inf), "soft": (0.0, 10.0)},
    "age": {"hard": (0.0, 130.0), "soft": (18.0, 100.0)},
    "late_30_59": {"hard": (0.0, np.inf), "soft": (0.0, 20.0)},
    "debt_ratio": {"hard": (0.0, np.inf), "soft": (0.0, 100.0)},
    "monthly_inc": {"hard": (0.0, np.inf), "soft": (0.0, 1e6)},
    "open_credit": {"hard": (0.0, np.inf), "soft": (0.0, 60.0)},
    "late_90": {"hard": (0.0, np.inf), "soft": (0.0, 20.0)},
    "real_estate": {"hard": (0.0, np.inf), "soft": (0.0, 30.0)},
    "late_60_89": {"hard": (0.0, np.inf), "soft": (0.0, 20.0)},
    "dependents": {"hard": (0.0, np.inf), "soft": (0.0, 15.0)},
}

FEATURES = list(COLUMN_SPEC)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Data validation failed:\n- " + "\n- ".join(self.errors))


def validate_frame(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    """Check columns, dtypes, value ranges and target coding.

    Parameters
    ----------
    df : DataFrame
        Raw benchmark data.
    require_target : bool
        Scoring data has no target, so the check can be switched off.
    """
    report = ValidationReport()
    expected = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        report.errors.append(f"missing columns: {missing}")

    for col in [c for c in FEATURES if c in df.columns]:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            report.errors.append(f"{col}: expected numeric dtype, got {series.dtype}")
            continue
        n_null = int(series.isna().sum())
        if n_null:
            report.warnings.append(f"{col}: {n_null} missing values")
        lo, hi = COLUMN_SPEC[col]["hard"]
        n_bad = int(((series < lo) | (series > hi)).sum())
        if n_bad:
            report.errors.append(f"{col}: {n_bad} values outside [{lo}, {hi}]")
        lo, hi = COLUMN_SPEC[col]["soft"]
        n_odd = int(((series < lo) | (series > hi)).sum())
        if n_odd:
            report.warnings.append(f"{col}: {n_odd} values outside [{lo}, {hi}]")

    if require_target and TARGET in df.columns:
        values = set(pd.unique(df[TARGET].dropna()))
        if not values <= {0, 1}:
            report.errors.append(f"{TARGET}: expected binary 0/1, got {sorted(values)}")

    n_dup = int(df.duplicated().sum())
    if n_dup:
        report.warnings.append(f"{n_dup} duplicated rows")
    return report


def assert_no_target_in_features(columns: list[str]) -> None:
    if TARGET in columns:
        raise ValueError(f"Target '{TARGET}' found in feature columns")
