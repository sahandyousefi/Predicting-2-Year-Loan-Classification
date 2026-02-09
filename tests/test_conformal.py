import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from credit_risk.evaluation.calibration import expected_calibration_error
from credit_risk.models.conformal import (
    MondrianConformalClassifier,
    VennAbersCalibrator,
    set_summary,
    three_way_decision,
)


def _sample(n, seed, signal=1.5):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n)
    z = rng.normal(0, 1, n) + signal * y
    return y, 1 / (1 + np.exp(-(z - 1)))


def test_label_conditional_coverage():
    y_cal, p_cal = _sample(2000, 0)
    y_te, p_te = _sample(4000, 1)
    cp = MondrianConformalClassifier().fit(y_cal, p_cal)
    summary = set_summary(cp.predict_set(p_te, alpha=0.1), y_te)
    assert summary["coverage_0"] >= 0.88
    assert summary["coverage_1"] >= 0.88
    shares = ["share_empty", "share_only_0", "share_only_1", "share_both"]
    assert sum(summary[k] for k in shares) == pytest.approx(1.0)


def test_p_values_are_valid():
    y_cal, p_cal = _sample(500, 2)
    pv = MondrianConformalClassifier().fit(y_cal, p_cal).p_values([0.0, 0.5, 1.0])
    assert ((pv > 0) & (pv <= 1)).all()
    assert pv[0, 0] == 1.0 and pv[2, 1] == 1.0


def test_uninformative_scores_give_ambiguous_sets():
    rng = np.random.default_rng(3)
    y_cal = rng.binomial(1, 0.5, 1000)
    cp = MondrianConformalClassifier().fit(y_cal, rng.uniform(size=1000))
    sets = cp.predict_set(rng.uniform(size=1000), alpha=0.05)
    assert set_summary(sets)["share_both"] > 0.8


def test_three_way_decision():
    sets = np.array([[True, False], [False, True], [True, True], [False, False]])
    assert list(three_way_decision(sets)) == ["approve", "decline", "refer", "refer"]


def test_fit_requires_both_classes():
    with pytest.raises(ValueError):
        MondrianConformalClassifier().fit([0, 0, 0], [0.1, 0.2, 0.3])
    cp = MondrianConformalClassifier().fit([0, 1], [0.1, 0.9])
    with pytest.raises(ValueError):
        cp.predict_set([0.5], alpha=0)


def test_venn_abers_interval_brackets_and_orders():
    y_cal, p_cal = _sample(400, 4)
    _, p_te = _sample(300, 5)
    res = VennAbersCalibrator().fit(y_cal, p_cal).predict_interval(p_te)
    assert (res.p0 <= res.p1 + 1e-12).all()
    assert ((res.p0 >= 0) & (res.p1 <= 1)).all()
    assert ((res.merged >= res.p0 - 1e-12) & (res.merged <= res.p1 + 1e-12)).all()
    assert list(res.to_frame().columns) == ["p0", "p1", "pd", "width"]


def test_venn_abers_matches_direct_isotonic_fit():
    y_cal, p_cal = _sample(150, 6)
    va = VennAbersCalibrator().fit(y_cal, p_cal)
    s = 0.37
    res = va.predict_interval([s])
    for label, value in ((0, res.p0[0]), (1, res.p1[0])):
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(np.append(p_cal, s), np.append(y_cal, label))
        assert value == pytest.approx(iso.predict([s])[0])


def test_venn_abers_width_shrinks_with_calibration_size():
    _, p_te = _sample(200, 7)
    widths = []
    for n in (100, 2000):
        y_cal, p_cal = _sample(n, 8)
        widths.append(VennAbersCalibrator().fit(y_cal, p_cal).predict_interval(p_te))
    assert widths[1].width.mean() < widths[0].width.mean()


def test_venn_abers_recalibrates_distorted_scores():
    y_cal, p_cal = _sample(2000, 9)
    y_te, p_te = _sample(2000, 10)
    va = VennAbersCalibrator().fit(y_cal, p_cal**3)
    merged = va.predict_proba(p_te**3)
    assert expected_calibration_error(y_te, merged) < 0.03
    assert expected_calibration_error(y_te, p_te**3) > 0.1


def test_uncertainty_report_and_cli(tmp_path, capsys):
    from credit_risk.cli import main
    from credit_risk.models.conformal import uncertainty_report

    y_cal, p_cal = _sample(600, 11)
    y_te, p_te = _sample(600, 12)
    sets, va = uncertainty_report(y_cal, p_cal, y_te, p_te, (0.1, 0.2))
    assert list(sets["alpha"]) == [0.1, 0.2]
    assert (sets["mean_size"].diff().dropna() <= 0).all()
    assert va["mean_width"] > 0

    train, holdout = tmp_path / "train.csv", tmp_path / "holdout.csv"
    main(["synthesize", str(train), "--n", "600", "--seed", "1"])
    main(["synthesize", str(holdout), "--n", "600", "--seed", "2"])
    out_dir = tmp_path / "out"
    main(["train", "--data", str(train), "--output-dir", str(out_dir)])
    capsys.readouterr()
    model = str(out_dir / "model.joblib")
    assert main(["conformal", str(holdout), "--model", model]) == 0
    assert "coverage_1" in capsys.readouterr().out
