import json

import pandas as pd

from credit_risk.cli import main


def test_synthesize_then_validate(tmp_path, capsys):
    path = tmp_path / "s.csv"
    assert main(["synthesize", str(path), "--n", "300"]) == 0
    assert len(pd.read_csv(path)) == 300
    assert main(["validate", str(path)]) == 0
    assert "0 errors" in capsys.readouterr().out


def test_validate_fails_on_bad_file(tmp_path, small_df):
    path = tmp_path / "bad.csv"
    small_df.drop(columns=["age"]).to_csv(path, index=False)
    assert main(["validate", str(path)]) == 1


def test_train_command(tmp_path, capsys):
    data = tmp_path / "s.csv"
    main(["synthesize", str(data), "--n", "600"])
    capsys.readouterr()
    out_dir = tmp_path / "out"
    assert main(["train", "--data", str(data), "--output-dir", str(out_dir)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "cv_roc_auc_mean" in report
    assert (out_dir / "metrics.json").exists()
