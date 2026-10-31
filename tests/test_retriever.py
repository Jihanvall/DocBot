import pytest

from src.retriever import Retriever

CHUNKS = [
    {"source": "a.md", "position": 0, "text": "the upload size limit is 20 megabytes"},
    {"source": "a.md", "position": 1, "text": "docker builds the backend image"},
    {"source": "b.md", "position": 0, "text": "the robot reaches the coffee cup"},
]


def test_best_match_comes_first():
    results = Retriever(CHUNKS).search("what is the upload size limit", k=2)
    assert results[0]["source"] == "a.md"
    assert results[0]["position"] == 0


def test_results_keep_source_position_and_score():
    result = Retriever(CHUNKS).search("docker image", k=1)[0]
    assert set(result) == {"source", "position", "text", "score"}
    assert 0 < result["score"] <= 1


def test_k_limits_the_number_of_results():
    assert len(Retriever(CHUNKS).search("the", k=1)) <= 1


def test_unrelated_question_returns_nothing():
    assert Retriever(CHUNKS).search("zebra giraffe", k=3) == []


def test_empty_chunks_are_rejected():
    with pytest.raises(ValueError):
        Retriever([])
