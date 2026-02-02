import pytest

from credit_risk.data.io import load_dataset


def test_load_drops_index_column(tmp_path, small_df):
    path = tmp_path / "d.csv"
    small_df.to_csv(path)
    df = load_dataset(path)
    assert not any(c.startswith("Unnamed") for c in df.columns)
    assert len(df) == len(small_df)


def test_load_raises_on_invalid_data(tmp_path, small_df):
    bad = small_df.copy()
    bad.loc[0, "late_90"] = -1
    path = tmp_path / "d.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_missing_file_points_to_readme(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_dataset(tmp_path / "nope.csv")
