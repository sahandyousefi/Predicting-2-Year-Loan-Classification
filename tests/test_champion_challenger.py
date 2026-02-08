import numpy as np
import pandas as pd
import pytest

from credit_risk.cli import main
from credit_risk.config import CostConfig
from credit_risk.models.persistence import ModelBundle, save_bundle
from credit_risk.monitoring.champion_challenger import (
    PromotionRule,
    champion_challenger,
    report_markdown,
    swap_set_table,
)

RULE = PromotionRule(n_boot=300)


@pytest.fixture(scope="module")
def outcomes():
    rng = np.random.default_rng(11)
    n = 3000
    true_pd = rng.beta(1.2, 5.0, n)
    y = (rng.uniform(size=n) < true_pd).astype(int)
    noisy = np.clip(true_pd + rng.normal(0, 0.15, n), 0.001, 0.999)
    return y, true_pd, noisy


def test_identical_models_are_not_promoted(outcomes):
    y, p, _ = outcomes
    res = champion_challenger(y, p, p, 0.15, 0.15, rule=RULE)
    diff = res.differences.set_index("metric")
    assert diff.loc["roc_auc", "diff"] == pytest.approx(0.0, abs=1e-12)
    assert diff.loc["profit_per_applicant", "ci_low"] == 0.0
    assert not res.promote


def test_better_challenger_is_promoted(outcomes):
    y, true_pd, noisy = outcomes
    cost = CostConfig()
    t = cost.margin / (cost.margin + cost.lgd)
    res = champion_challenger(y, noisy, true_pd, t, t, cost, RULE)
    assert res.promote, res.rationale
    swap = res.swap_table.set_index("segment")
    assert swap.loc["swap_in", "bad_rate"] < swap.loc["swap_out", "bad_rate"]


def test_worse_challenger_is_rejected(outcomes):
    y, true_pd, noisy = outcomes
    res = champion_challenger(y, true_pd, noisy, 0.15, 0.15, rule=RULE)
    assert not res.promote
    assert any(line.startswith("fail") for line in res.rationale)


def test_swap_table_partitions_population():
    y = np.array([0, 1, 0, 1, 0])
    a = np.array([True, True, False, False, True])
    b = np.array([True, False, True, False, False])
    t = swap_set_table(y, a, b)
    assert t["count"].sum() == 5
    assert t["share"].sum() == pytest.approx(1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        champion_challenger([0, 1], [0.1, 0.2], [0.1], 0.5, 0.5)


def test_report_markdown(outcomes):
    y, p, q = outcomes
    md = report_markdown(champion_challenger(y, p, q, 0.2, 0.2, rule=RULE))
    assert "Decision:" in md and "swap_in" in md


def test_challenge_cli(tmp_path, capsys):
    data = tmp_path / "s.csv"
    main(["synthesize", str(data), "--n", "800", "--seed", "1"])
    main(["train", "--data", str(data), "--output-dir", str(tmp_path / "a")])
    champion = tmp_path / "a" / "model.joblib"

    from credit_risk.models.persistence import load_bundle

    base = load_bundle(champion)
    flat = ModelBundle(
        model=_ConstantModel(0.5),
        features=base.features,
        threshold=0.6,
        background=base.background,
    )
    challenger = save_bundle(flat, tmp_path / "flat.joblib")
    holdout = tmp_path / "h.csv"
    main(["synthesize", str(holdout), "--n", "600", "--seed", "2"])
    report = tmp_path / "cc.md"
    rc = main(
        [
            "challenge",
            str(holdout),
            "--champion",
            str(champion),
            "--challenger",
            str(challenger),
            "--n-boot",
            "200",
            "--output",
            str(report),
        ]
    )
    assert rc == 0
    assert "retain champion" in report.read_text()


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.value)
        return np.column_stack([1 - p, p])
