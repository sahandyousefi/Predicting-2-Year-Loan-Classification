import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk.features.engineering import build_xy
from credit_risk.models.neural import PiecewiseLinearEncoder


def test_ple_bounds_and_monotonicity():
    x = np.linspace(-3, 3, 200)[:, None]
    enc = PiecewiseLinearEncoder(n_bins=8).fit(x)
    Z = enc.transform(x)
    assert Z.shape == (200, 8)
    assert Z.min() >= 0 and Z.max() <= 1
    # The sum of components is a monotone function of x.
    assert (np.diff(Z.sum(axis=1)) >= 0).all()
    assert Z[0].sum() == 0 and Z[-1].sum() == 8


def test_ple_merges_duplicate_edges_for_counts():
    x = np.repeat([0.0, 1.0, 2.0], [80, 15, 5])[:, None]
    enc = PiecewiseLinearEncoder(n_bins=16).fit(x)
    assert len(enc.edges_[0]) <= 3
    with pytest.raises(ValueError):
        enc.transform(np.zeros((2, 2)))


def test_ple_constant_column():
    enc = PiecewiseLinearEncoder().fit(np.ones((10, 1)))
    assert enc.transform(np.ones((3, 1))).shape == (3, 1)


@pytest.fixture(scope="module")
def split(credit_df):
    X, y = build_xy(credit_df)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


@pytest.mark.parametrize("encoding", ["ple", "standard"])
def test_mlp_fits_and_ranks(split, encoding):
    pytest.importorskip("torch")
    from credit_risk.models.neural import TabularMLP

    X_tr, X_te, y_tr, y_te = split
    model = TabularMLP(encoding=encoding, max_epochs=30, patience=5).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert roc_auc_score(y_te, proba[:, 1]) > 0.65
    assert 1 <= model.best_epoch_ <= len(model.history_)


def test_mlp_is_reproducible_and_picklable(split, tmp_path):
    pytest.importorskip("torch")
    import joblib

    from credit_risk.models.neural import TabularMLP

    X_tr, X_te, y_tr, _ = split
    a = TabularMLP(max_epochs=5, random_state=3).fit(X_tr, y_tr)
    b = TabularMLP(max_epochs=5, random_state=3).fit(X_tr, y_tr)
    assert np.allclose(a.predict_proba(X_te), b.predict_proba(X_te))
    path = tmp_path / "mlp.joblib"
    joblib.dump(a, path)
    restored = joblib.load(path)
    assert np.allclose(restored.predict_proba(X_te), a.predict_proba(X_te))


def test_mlp_rejects_unknown_encoding(split):
    pytest.importorskip("torch")
    from credit_risk.models.neural import TabularMLP

    X_tr, _, y_tr, _ = split
    with pytest.raises(ValueError):
        TabularMLP(encoding="onehot").fit(X_tr, y_tr)
