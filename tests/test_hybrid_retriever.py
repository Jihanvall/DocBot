import pytest

from src.hybrid_retriever import HybridRetriever


class FakeRetriever:
    def __init__(self, order):
        self.order = order

    def search(self, question, k=3):
        return [{"source": "a.md", "position": p, "text": f"chunk {p}", "score": 1.0} for p in self.order[:k]]


def test_chunk_found_by_both_methods_wins():
    hybrid = HybridRetriever([FakeRetriever([0, 1, 2]), FakeRetriever([2, 0, 1])])
    assert hybrid.search("q", k=1)[0]["position"] == 0


def test_chunk_found_by_only_one_method_is_still_returned():
    hybrid = HybridRetriever([FakeRetriever([0]), FakeRetriever([5])])
    positions = {h["position"] for h in hybrid.search("q", k=2)}
    assert positions == {0, 5}


def test_k_limits_results():
    hybrid = HybridRetriever([FakeRetriever([0, 1, 2]), FakeRetriever([3, 4, 5])])
    assert len(hybrid.search("q", k=2)) == 2


def test_results_have_the_standard_fields():
    hybrid = HybridRetriever([FakeRetriever([0])])
    assert set(hybrid.search("q", k=1)[0]) == {"source", "position", "text", "score"}


def test_empty_retrievers_rejected():
    with pytest.raises(ValueError):
        HybridRetriever([])
