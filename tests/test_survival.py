import numpy as np
import pytest

from credit_risk.cli import main
from credit_risk.data.survival import (
    DURATION,
    EVENT,
    SurvivalSpec,
    binary_label,
    simulate_survival,
)
from credit_risk.features.engineering import FEATURE_COLUMNS, add_engineered_features
from credit_risk.models.survival import (
    DiscreteTimeHazard,
    concordance_index,
    kaplan_meier,
    person_period,
    survival_study,
)


@pytest.fixture(scope="module")
def sample():
    return simulate_survival(SurvivalSpec(n=4000, seed=1))


def _x(frame):
    return add_engineered_features(frame)[FEATURE_COLUMNS]


def test_simulation_hits_target_pd_and_window(sample):
    f = sample.frame
    assert sample.true_cumulative_pd(24).mean() == pytest.approx(0.15, abs=1e-6)
    assert f[DURATION].between(1, 36).all()
    assert set(f[EVENT].unique()) <= {0, 1}
    assert (f.loc[f[EVENT] == 0, DURATION] <= 36).all()


def test_binary_label_known_mask(sample):
    y, known = binary_label(sample.frame, 24)
    f = sample.frame
    assert (y[known == 0] == 0).all()
    censored_early = (f[EVENT] == 0) & (f[DURATION] < 24)
    assert not known[censored_early.to_numpy()].any()


def test_person_period_rows_and_labels():
    rows, period, y = person_period(np.zeros((3, 1)), [2, 1, 3], [1, 0, 0])
    assert rows.tolist() == [0, 0, 1, 2, 2, 2]
    assert period.tolist() == [1, 2, 1, 1, 2, 3]
    assert y.tolist() == [0, 1, 0, 0, 0, 0]


def test_person_period_rejects_zero_duration():
    with pytest.raises(ValueError):
        person_period(np.zeros((1, 1)), [0], [0])


def test_kaplan_meier_matches_hand_computation():
    # Month 1: 4 at risk, 1 default. Month 2: 2 at risk (one censored), 1 default.
    s = kaplan_meier(np.array([1, 1, 2, 3]), np.array([1, 0, 1, 0]), 3)
    assert s == pytest.approx([0.75, 0.375, 0.375])


def test_concordance_index_extremes():
    d = np.array([1, 2, 3, 4])
    e = np.array([1, 1, 1, 0])
    assert concordance_index(d, e, -d.astype(float)) == 1.0
    assert concordance_index(d, e, d.astype(float)) == 0.0
    assert concordance_index(d, e, np.zeros(4)) == 0.5


def test_hazard_model_recovers_true_cumulative_pd(sample):
    f = sample.frame
    model = DiscreteTimeHazard().fit(_x(f), f[DURATION], f[EVENT])
    for h in (12, 24, 36):
        pred = model.cumulative_pd(_x(f), h)
        truth = sample.true_cumulative_pd(h)
        assert pred.mean() == pytest.approx(truth.mean(), abs=0.01)
        assert np.corrcoef(pred, truth)[0, 1] > 0.95


def test_survival_function_is_non_increasing(sample):
    f = sample.frame.head(500)
    model = DiscreteTimeHazard(time_bins=(6, 12, 24)).fit(_x(f), f[DURATION], f[EVENT])
    s = model.survival(_x(f))
    assert s.shape == (500, 24)
    assert (np.diff(s, axis=1) <= 1e-12).all()
    with pytest.raises(ValueError):
        model.cumulative_pd(_x(f), 30)


def test_invalid_time_bins():
    with pytest.raises(ValueError):
        DiscreteTimeHazard(time_bins=(6, 3))


def test_study_shows_censoring_bias():
    s = simulate_survival(SurvivalSpec(n=6000, censor_rate=0.03, seed=2))
    table = survival_study(
        s.frame, {12: s.true_cumulative_pd(12), 24: s.true_cumulative_pd(24)}
    ).set_index(["model", "horizon"])
    hazard = table.loc[("Discrete-time hazard", 24)]
    dropped = table.loc[("Binary, drop censored", 24)]
    as_good = table.loc[("Binary, censored as good", 24)]
    assert abs(hazard.mean_pd - hazard.true_mean_pd) < 0.01
    assert dropped.mean_pd > hazard.true_mean_pd + 0.02
    assert as_good.mean_pd < hazard.true_mean_pd - 0.02
    assert hazard.mae_vs_true < min(dropped.mae_vs_true, as_good.mae_vs_true)
    assert ("Discrete-time hazard", 12) in table.index


def test_cli_survival(tmp_path, capsys):
    out = tmp_path / "survival.csv"
    code = main(
        ["survival", "--n", "1500", "--horizons", "12", "24", "--output", str(out)]
    )
    assert code == 0
    assert out.exists()
    assert "Discrete-time hazard" in capsys.readouterr().out
