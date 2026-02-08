"""Dense features from short free-text fields.

Two backends share one scikit-learn interface:

``tfidf``
    Word unigram and bigram TF-IDF followed by truncated SVD (latent semantic
    analysis). Needs only scikit-learn and is used in tests.
``sbert``
    Mean-pooled Sentence-BERT embeddings (``embed`` extra), optionally reduced
    with PCA fitted on the training rows.

The reduction is fitted inside the estimator, so cross-validation never sees
test-fold vocabulary or components.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

BACKENDS = ("tfidf", "sbert")


def _as_texts(X: Iterable) -> list[str]:
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError("expected a single text column")
        X = X.iloc[:, 0]
    arr = np.asarray(X, dtype=object).ravel()
    return ["" if pd.isna(v) else str(v) for v in arr]


class TextEmbedder(TransformerMixin, BaseEstimator):
    """Map a text column to ``n_components`` dense features.

    Parameters
    ----------
    backend : {"tfidf", "sbert"}
        Embedding method.
    n_components : int
        Output dimension. For ``sbert`` a value of 0 keeps the full encoder
        dimension.
    model_name : str
        Sentence-Transformers model for the ``sbert`` backend.
    random_state : int
        Seed of the SVD or PCA solver.
    """

    def __init__(
        self,
        backend: str = "tfidf",
        n_components: int = 16,
        model_name: str = "all-MiniLM-L6-v2",
        random_state: int = 0,
    ) -> None:
        self.backend = backend
        self.n_components = n_components
        self.model_name = model_name
        self.random_state = random_state

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.encoder_.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def fit(self, X, y=None):
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        texts = _as_texts(X)
        if self.backend == "tfidf":
            self.vectorizer_ = TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, sublinear_tf=True
            )
            tf = self.vectorizer_.fit_transform(texts)
            k = max(1, min(self.n_components, tf.shape[1] - 1))
            self.reducer_ = TruncatedSVD(n_components=k, random_state=self.random_state)
            self.reducer_.fit(tf)
        else:
            from sentence_transformers import SentenceTransformer

            self.encoder_ = SentenceTransformer(self.model_name)
            emb = self._encode(texts)
            self.reducer_ = None
            if self.n_components:
                self.reducer_ = PCA(
                    n_components=min(self.n_components, emb.shape[1]),
                    random_state=self.random_state,
                ).fit(emb)
        self.n_features_out_ = self._dimension()
        return self

    def _dimension(self) -> int:
        if self.reducer_ is not None:
            return int(self.reducer_.n_components)
        return int(self.encoder_.get_sentence_embedding_dimension())

    def transform(self, X) -> np.ndarray:
        texts = _as_texts(X)
        if self.backend == "tfidf":
            z = self.reducer_.transform(self.vectorizer_.transform(texts))
            return normalize(z)
        emb = self._encode(texts)
        return emb if self.reducer_ is None else self.reducer_.transform(emb)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([f"txt_{i}" for i in range(self.n_features_out_)], dtype=object)
