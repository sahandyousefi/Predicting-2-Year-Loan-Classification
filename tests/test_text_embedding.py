import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from credit_risk.data.schema import TARGET
from credit_risk.data.synthetic_text import (
    TEXT_COLUMN,
    add_purpose_text,
    make_purpose_text,
)
from credit_risk.features.text_embedding import TextEmbedder


def test_purpose_text_is_reproducible(small_df):
    a = add_purpose_text(small_df, seed=1)
    b = add_purpose_text(small_df, seed=1)
    assert a[TEXT_COLUMN].equals(b[TEXT_COLUMN])
    assert a[TEXT_COLUMN].str.len().gt(0).all()
    assert list(a.columns[:-1]) == list(small_df.columns)


def test_purpose_text_requires_target(small_df):
    with pytest.raises(ValueError):
        add_purpose_text(small_df.drop(columns=[TARGET]))


def _risk_share(texts: pd.Series) -> float:
    return texts.str.contains("past due|late bills|behind on|urgent|fast").mean()


def test_signal_controls_dependence():
    y = np.repeat([0, 1], 3000)
    strong = make_purpose_text(y, signal=1.0, seed=0)
    null = make_purpose_text(y, signal=0.0, seed=0)
    gap_strong = _risk_share(strong[y == 1]) - _risk_share(strong[y == 0])
    gap_null = _risk_share(null[y == 1]) - _risk_share(null[y == 0])
    assert gap_strong > 0.05
    assert abs(gap_null) < 0.03


def test_tfidf_embedder_shape_and_names(small_df):
    texts = add_purpose_text(small_df)[TEXT_COLUMN]
    emb = TextEmbedder(n_components=8).fit(texts)
    z = emb.transform(texts)
    assert z.shape == (len(texts), 8)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-6)
    assert list(emb.get_feature_names_out()) == [f"txt_{i}" for i in range(8)]


def test_embedder_accepts_frame_and_missing_values():
    df = pd.DataFrame({TEXT_COLUMN: ["pay off cards", None, "new roof"] * 10})
    z = clone(TextEmbedder(n_components=3)).fit_transform(df)
    assert z.shape == (30, 3)
    assert np.isfinite(z).all()


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        TextEmbedder(backend="bag").fit(["a b", "b c"])


def test_sbert_backend():
    pytest.importorskip("sentence_transformers")
    try:
        emb = TextEmbedder(backend="sbert", n_components=4).fit(
            ["new roof", "pay overdue rent", "car purchase", "medical bills"] * 3
        )
    except OSError:
        pytest.skip("encoder weights not available offline")
    assert emb.transform(["wedding costs"]).shape == (1, 4)
