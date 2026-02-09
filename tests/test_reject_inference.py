import numpy as np
import pytest

from credit_risk.cli import main
from credit_risk.data.schema import TARGET
from credit_risk.data.through_the_door import (
    ACCEPTED,
    ThroughTheDoorSpec,
    make_through_the_door,
)
from credit_risk.features.engineering import FEATURE_COLUMNS, add_engineered_features
from credit_risk.models.reject_inference import (
    METHODS,
    acceptance_weights,
    fit_reject_inference,
    fuzzy_augmentation,
    parceling,
    reject_inference_study,
)


@pytest.fixture(scope="module")
def ttd():
    df = make_through_the_door(ThroughTheDoorSpec(n=6000, hidden_signal=0.5, seed=1))
    fe = add_engineered_features(df)
    return (
        fe[FEATURE_COLUMNS],
        fe[TARGET].to_numpy(),
        fe[ACCEPTED].to_numpy(),
        df,
    )


def test_generator_applies_policy(ttd):
    _, y, acc, _ = ttd
    assert acc.mean() == pytest.approx(0.6, abs=0.001)
    assert y[~acc].mean() > 2 * y[acc].mean()


def test_spec_validation():
    with pytest.raises(ValueError):
        ThroughTheDoorSpec(approval_rate=1.0)
    with pytest.raises(ValueError):
        ThroughTheDoorSpec(hidden_signal=-1)


def test_fuzzy_weights(ttd):
    X, y, acc, _ = ttd
    kgb = fit_reject_inference("kgb", X, y, acc)
    sample = fuzzy_augmentation(X[acc], y[acc], X[~acc], kgb)
    assert len(sample.X) == acc.sum() + 2 * (~acc).sum()
    assert sample.weight.sum() == pytest.approx(len(X))


def test_parceling_inflation_raises_reject_bad_rate(ttd):
    X, y, acc, _ = ttd
    kgb = fit_reject_inference("kgb", X, y, acc)
    n_acc = acc.sum()
    rates = [
        parceling(X[acc], y[acc], X[~acc], kgb, k, seed=0).y[n_acc:].mean()
        for k in (1.0, 2.0)
    ]
    assert rates[1] > 1.5 * rates[0]
    with pytest.raises(ValueError):
        parceling(X[acc], y[acc], X[~acc], kgb, 0.5)


def test_acceptance_weights_are_bounded(ttd):
    X, _, acc, _ = ttd
    w = acceptance_weights(X, acc, clip=10.0)
    assert len(w) == acc.sum()
    assert (w >= 1.0).all() and (w <= 10.0).all()


def test_rejected_outcomes_are_not_used(ttd):
    X, y, acc, _ = ttd
    scrambled = y.copy()
    scrambled[~acc] = 1 - scrambled[~acc]
    for method in METHODS:
        a = fit_reject_inference(method, X, y, acc).predict_proba(X)[:, 1]
        b = fit_reject_inference(method, X, scrambled, acc).predict_proba(X)[:, 1]
        assert np.allclose(a, b), method


def test_unknown_method(ttd):
    X, y, acc, _ = ttd
    with pytest.raises(ValueError):
        fit_reject_inference("em", X, y, acc)


def test_study_shows_kgb_optimism(ttd):
    *_, df = ttd
    table = reject_inference_study(df, inflation=2.0).set_index("method")
    assert list(table.index) == ["oracle", *METHODS]
    kgb, oracle = table.loc["kgb"], table.loc["oracle"]
    assert kgb["mean_pd_rejects"] < 0.6 * kgb["bad_rate_rejects"]
    assert oracle["auc_all"] > kgb["auc_all"]
    assert table.loc["parceling", "mean_pd_rejects"] > kgb["mean_pd_rejects"]


def test_reject_inference_cli(tmp_path, capsys):
    out = tmp_path / "ri.csv"
    rc = main(["reject-inference", "--n", "3000", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    assert "parceling" in capsys.readouterr().out
