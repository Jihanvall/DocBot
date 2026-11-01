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
