import numpy as np
import pytest

from src.embedding_retriever import EmbeddingRetriever

CHUNKS = [
    {"source": "a.md", "position": 0, "text": "upload limit"},
    {"source": "a.md", "position": 1, "text": "docker image"},
    {"source": "b.md", "position": 0, "text": "robot coffee"},
]

VECTORS = {
    "upload limit": [1.0, 0.0, 0.0],
    "docker image": [0.0, 1.0, 0.0],
    "robot coffee": [0.0, 0.0, 1.0],
    "how big can a file be": [0.9, 0.1, 0.0],
}


def fake_embed(texts, task_type=None):
    return np.array([VECTORS[t] for t in texts], dtype=float)


def test_closest_chunk_comes_first():
    r = EmbeddingRetriever(CHUNKS, embed=fake_embed)
    assert r.search("how big can a file be", k=1)[0]["text"] == "upload limit"


def test_results_keep_the_same_fields_as_tfidf():
    r = EmbeddingRetriever(CHUNKS, embed=fake_embed)
    result = r.search("how big can a file be", k=1)[0]
    assert set(result) == {"source", "position", "text", "score"}


def test_k_limits_results():
    r = EmbeddingRetriever(CHUNKS, embed=fake_embed)
    assert len(r.search("how big can a file be", k=2)) == 2


def test_empty_chunks_rejected():
    with pytest.raises(ValueError):
        EmbeddingRetriever([], embed=fake_embed)


def test_chunk_embeddings_are_cached_between_retrievers(tmp_path, monkeypatch):
    import src.embedding_cache as cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    calls = []

    def counting_embed(texts, task_type=None):
        calls.append(task_type)
        return fake_embed(texts, task_type)

    EmbeddingRetriever(CHUNKS, embed=counting_embed)
    EmbeddingRetriever(CHUNKS, embed=counting_embed)
    assert calls.count("RETRIEVAL_DOCUMENT") == 1


def test_repeated_question_does_not_call_the_embedder_again(tmp_path, monkeypatch):
    import src.embedding_cache as cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    calls = []

    def counting_embed(texts, task_type=None):
        calls.append(texts)
        return fake_embed(texts, task_type)

    r = EmbeddingRetriever(CHUNKS, embed=counting_embed)
    r.search("how big can a file be", k=1)
    r.search("how big can a file be", k=1)
    question_calls = [c for c in calls if c == ["how big can a file be"]]
    assert len(question_calls) == 1
