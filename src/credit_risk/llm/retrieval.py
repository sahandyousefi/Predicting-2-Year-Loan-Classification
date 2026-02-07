from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

POLICY_DIR = Path(__file__).parent / "policy"
EVAL_FILE = Path(__file__).parent / "eval" / "retrieval_queries.yaml"


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    source: str


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """One chunk per second-level section whose heading starts with an id."""
    chunks = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        heading, _, body = block.partition("\n")
        match = re.match(r"(\S+)\s+(.*)", heading.strip())
        if match:
            chunks.append(Chunk(match.group(1), match.group(2), body.strip(), source))
    return chunks


def load_policy_corpus(directory: Path = POLICY_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(), path.name))
    return chunks


class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]: ...


class TfidfRetriever:
    """Sparse lexical retrieval on character n-grams, robust to word forms."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(
            [f"{c.title}. {c.text}" for c in chunks]
        )

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class EmbeddingRetriever:
    """Dense retrieval with a Sentence-BERT model (``embed`` extra)."""

    def __init__(self, chunks: list[Chunk], model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.encoder = SentenceTransformer(model)
        self.matrix = self.encoder.encode(
            [f"{c.title}. {c.text}" for c in chunks], normalize_embeddings=True
        )

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self.encoder.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


def load_eval_queries(path: Path = EVAL_FILE) -> list[dict]:
    return yaml.safe_load(path.read_text())


def evaluate_retriever(
    retriever: Retriever, queries: list[dict], k: int = 3
) -> dict[str, float]:
    """Recall@k (share of relevant ids found in the top k) and MRR."""
    recalls, rr = [], []
    for item in queries:
        relevant = set(item["relevant"])
        ranked = [c.id for c, _ in retriever.search(item["query"], k=len(relevant) + 9)]
        recalls.append(len(relevant & set(ranked[:k])) / len(relevant))
        first = next((i for i, cid in enumerate(ranked, 1) if cid in relevant), None)
        rr.append(1 / first if first else 0.0)
    return {f"recall@{k}": float(np.mean(recalls)), "mrr": float(np.mean(rr))}
