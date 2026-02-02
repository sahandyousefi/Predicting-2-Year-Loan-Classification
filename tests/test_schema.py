import pytest

from credit_risk.data.schema import (
    TARGET,
    assert_no_target_in_features,
    validate_frame,
)
from tests.test_pipeline import make_frame


def test_clean_frame_passes():
    report = validate_frame(make_frame(100))
    assert report.ok, report.errors


def test_missing_column_is_error():
    report = validate_frame(make_frame(20).drop(columns=["age"]))
    assert not report.ok
    with pytest.raises(ValueError):
        report.raise_for_errors()


def test_negative_income_is_error():
    df = make_frame(20)
    df.loc[0, "monthly_inc"] = -5
    assert not validate_frame(df).ok


def test_unusual_age_is_only_warning():
    df = make_frame(20)
    df.loc[0, "age"] = 105
    report = validate_frame(df)
    assert report.ok and any("age" in w for w in report.warnings)


def test_non_binary_target_is_error():
    df = make_frame(20)
    df.loc[0, TARGET] = 2
    assert not validate_frame(df).ok


def test_scoring_frame_without_target():
    assert validate_frame(make_frame(20).drop(columns=[TARGET]), False).ok


def test_target_in_features_rejected():
    with pytest.raises(ValueError):
        assert_no_target_in_features(["age", TARGET])
