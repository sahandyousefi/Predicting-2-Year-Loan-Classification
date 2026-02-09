from sklearn.model_selection import train_test_split

from credit_risk.cli import main
from credit_risk.evaluation.compare import compare_models, to_markdown
from credit_risk.features.engineering import build_xy


def test_compare_table(credit_df):
    X, y = build_xy(credit_df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    table = compare_models(
        ["logistic", "scorecard", "hist_gbm"], X_tr, y_tr, X_te, y_te
    )
    assert list(table["model"]) and len(table) == 3
    assert table["roc_auc"].is_monotonic_decreasing
    assert table["delong_p_vs_best"].isna().sum() == 1
    md = to_markdown(table)
    assert md.count("\n") == 4


def test_compare_cli(tmp_path, capsys):
    data = tmp_path / "s.csv"
    main(["synthesize", str(data), "--n", "800"])
    rc = main(
        [
            "compare",
            "--data",
            str(data),
            "--output-dir",
            str(tmp_path),
            "--models",
            "logistic",
            "scorecard",
        ]
    )
    assert rc == 0
    assert (tmp_path / "model_comparison.md").exists()


def test_compare_monotone_cli(tmp_path):
    data = tmp_path / "s.csv"
    main(["synthesize", str(data), "--n", "800"])
    rc = main(
        [
            "compare",
            "--data",
            str(data),
            "--output-dir",
            str(tmp_path),
            "--models",
            "logistic",
            "hist_gbm",
            "--monotone",
        ]
    )
    assert rc == 0
