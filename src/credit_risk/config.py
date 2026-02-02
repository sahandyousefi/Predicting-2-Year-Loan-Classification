from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class DataConfig:
    path: str = "data/Credit Risk Benchmark Dataset.csv"
    target: str = "dlq_2yrs"
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("data.test_size must be in (0, 1)")


@dataclass(frozen=True)
class ModelConfig:
    C: float = 100.0
    l1_ratio: float = 1.0
    cv_folds: int = 5

    def __post_init__(self) -> None:
        if self.C <= 0:
            raise ValueError("model.C must be positive")
        if self.cv_folds < 2:
            raise ValueError("model.cv_folds must be at least 2")


@dataclass(frozen=True)
class CostConfig:
    lgd: float = 0.45
    ead: float = 1.0
    margin: float = 0.08

    def __post_init__(self) -> None:
        if not 0.0 <= self.lgd <= 1.0:
            raise ValueError("cost.lgd must be in [0, 1]")
        if self.ead <= 0 or self.margin < 0:
            raise ValueError("cost.ead must be positive and cost.margin non-negative")


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    output_dir: str = "artifacts"


def _build(cls: type, values: dict[str, Any] | None) -> Any:
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config, falling back to dataclass defaults for missing keys."""
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}
    unknown = set(raw) - {"data", "model", "cost", "output_dir"}
    if unknown:
        raise ValueError(f"Unknown top-level keys: {sorted(unknown)}")
    return Config(
        data=_build(DataConfig, raw.get("data")),
        model=_build(ModelConfig, raw.get("model")),
        cost=_build(CostConfig, raw.get("cost")),
        output_dir=raw.get("output_dir", "artifacts"),
    )
