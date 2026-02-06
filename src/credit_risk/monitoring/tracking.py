from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from credit_risk import __version__


def _sha(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def data_hash(df: pd.DataFrame) -> str:
    """Order-sensitive content hash of a DataFrame."""
    values = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_fingerprint(config: Any, df: pd.DataFrame) -> dict[str, Any]:
    """Identifiers that make a training run reproducible and auditable."""
    cfg = asdict(config) if is_dataclass(config) else config
    return {
        "config_hash": _sha(cfg),
        "data_hash": data_hash(df),
        "n_rows": int(len(df)),
        "package_version": __version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "git_commit": git_commit(),
    }


def log_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    output_dir: str | Path,
    artifacts: list[str | Path] | None = None,
    experiment: str = "credit-risk",
    use_mlflow: bool | None = None,
) -> str:
    """Record a run in MLflow when available, else as a JSON file.

    Returns the run identifier.
    """
    if use_mlflow is None:
        use_mlflow = importlib.util.find_spec("mlflow") is not None
    if use_mlflow:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run() as run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(metrics)
            for path in artifacts or []:
                mlflow.log_artifact(str(path))
            return run.info.run_id

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    runs = Path(output_dir) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "experiment": experiment,
        "params": params,
        "metrics": metrics,
        "artifacts": [str(a) for a in artifacts or []],
    }
    (runs / f"{run_id}.json").write_text(json.dumps(record, indent=2, default=str))
    return run_id
