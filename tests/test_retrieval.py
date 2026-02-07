import pytest

from credit_risk.llm.retrieval import (
    TfidfRetriever,
    chunk_markdown,
    evaluate_retriever,
    load_eval_queries,
    load_policy_corpus,
)


@pytest.fixture(scope="module")
def corpus():
    return load_policy_corpus()


def test_chunking():
    text = "# T\nintro\n## A-1 First\nbody one\n## A-2 Second\nbody two\n"
    chunks = chunk_markdown(text, "x.md")
    assert [c.id for c in chunks] == ["A-1", "A-2"]
    assert chunks[1].title == "Second" and chunks[1].text == "body two"


def test_corpus_ids_unique(corpus):
    ids = [c.id for c in corpus]
    assert len(ids) == len(set(ids)) >= 10


def test_eval_queries_reference_existing_ids(corpus):
    ids = {c.id for c in corpus}
    for q in load_eval_queries():
        assert set(q["relevant"]) <= ids


def test_tfidf_retrieval_quality(corpus):
    # Regression guard on the labeled query set.
    metrics = evaluate_retriever(TfidfRetriever(corpus), load_eval_queries(), k=3)
    assert metrics["recall@3"] >= 0.8
    assert metrics["mrr"] >= 0.75


def test_embedding_retriever(corpus):
    pytest.importorskip("sentence_transformers")
    from credit_risk.llm.retrieval import EmbeddingRetriever

    metrics = evaluate_retriever(EmbeddingRetriever(corpus), load_eval_queries(), k=3)
    assert metrics["recall@3"] >= 0.7
