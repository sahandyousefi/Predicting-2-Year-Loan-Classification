from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from credit_risk.data.schema import validate_frame

logger = logging.getLogger(__name__)


def load_dataset(
    path: str | Path, validate: bool = True, require_target: bool = True
) -> pd.DataFrame:
    """Read the benchmark CSV and run schema validation.

    Validation errors raise; warnings are logged so that unusual but legal
    values (for example ages above 100) do not block a run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for download instructions."
        )
    df = pd.read_csv(path)
    # Kaggle exports sometimes carry the pandas index as an unnamed column.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if validate:
        report = validate_frame(df, require_target=require_target)
        for warning in report.warnings:
            logger.warning(warning)
        report.raise_for_errors()
    return df
