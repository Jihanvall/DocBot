from src.evaluation import contains_keywords, hit_at_k, is_refusal, summarize

HITS = [{"source": "a.md"}, {"source": "b.md"}]


def test_hit_when_expected_source_is_retrieved():
    assert hit_at_k(HITS, "b.md") is True
    assert hit_at_k(HITS, "c.md") is False


def test_keywords_ignore_case_and_need_all():
    assert contains_keywords("The limit is 20MB", ["20", "mb"]) is True
    assert contains_keywords("The limit is 20", ["20", "mb"]) is False


def test_refusal_detection():
    assert is_refusal("I don't know based on the provided documents.") is True
    assert is_refusal("It is 20 MB.") is False


def test_summary_computes_the_three_rates():
    rows = [
        {"expected_source": "a", "hit": True, "correct": True, "refused": False},
        {"expected_source": "a", "hit": True, "correct": False, "refused": False},
        {"expected_source": None, "hit": False, "correct": False, "refused": True},
    ]
    assert summarize(rows) == {
        "questions": 3,
        "hit_rate": 1.0,
        "keyword_accuracy": 0.5,
        "correct_refusals": 1.0,
    }


def test_summary_handles_missing_groups():
    rows = [{"expected_source": "a", "hit": True, "correct": True, "refused": False}]
    assert summarize(rows)["correct_refusals"] is None
