import json

from credit_risk.config import load_config
from credit_risk.monitoring.tracking import data_hash, log_run, run_fingerprint


def test_fingerprint_is_stable(small_df):
    cfg = load_config()
    a = run_fingerprint(cfg, small_df)
    b = run_fingerprint(cfg, small_df.copy())
    assert a["config_hash"] == b["config_hash"]
    assert a["data_hash"] == b["data_hash"]


def test_data_hash_detects_change(small_df):
    changed = small_df.copy()
    changed.loc[0, "age"] += 1
    assert data_hash(changed) != data_hash(small_df)


def test_json_fallback(tmp_path):
    run_id = log_run({"C": 1.0}, {"auc": 0.8}, tmp_path, use_mlflow=False)
    record = json.loads((tmp_path / "runs" / f"{run_id}.json").read_text())
    assert record["metrics"]["auc"] == 0.8
