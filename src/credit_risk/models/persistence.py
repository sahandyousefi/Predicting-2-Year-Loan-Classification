from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_risk import __version__


@dataclass
class ModelBundle:
    """Everything needed to score and explain new applicants."""

    model: Any
    features: list[str]
    threshold: float
    background: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    package_version: str = __version__


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle")
    return bundle
