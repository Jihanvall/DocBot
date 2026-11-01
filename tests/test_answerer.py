import pytest

from src import answerer
from src.answerer import NO_ANSWER, AnswerError, answer_question, build_prompt

HITS = [
    {
        "source": "a.md",
        "position": 0,
        "text": "the upload limit is 20 megabytes",
        "score": 0.9,
    },
    {"source": "b.md", "position": 2, "text": "docker builds the image", "score": 0.3},
]


class FakeResponse:
    text = "  The limit is 20 MB [1].  "


class FakeModels:
    def generate_content(self, **kwargs):
        return FakeResponse()


class FakeClient:
    models = FakeModels()


def test_no_hits_returns_dont_know_without_calling_gemini(monkeypatch):
    def boom(api_key=None):
        raise AssertionError("Gemini must not be called")

    monkeypatch.setattr(answerer.genai, "Client", boom)
    assert answer_question("anything", []) == {"answer": NO_ANSWER, "sources": []}


def test_answer_is_stripped_and_sources_are_returned(monkeypatch):
    monkeypatch.setattr(answerer.genai, "Client", lambda api_key=None: FakeClient())
    result = answer_question("what is the limit?", HITS)
    assert result["answer"] == "The limit is 20 MB [1]."
    assert result["sources"] == [
        {"source": "a.md", "position": 0},
        {"source": "b.md", "position": 2},
    ]


def test_gemini_failure_raises_a_clear_error(monkeypatch):
    def boom(api_key=None):
        raise RuntimeError("no network")

    monkeypatch.setattr(answerer.genai, "Client", boom)
    with pytest.raises(AnswerError):
        answer_question("q", HITS)


def test_empty_answer_raises_a_clear_error(monkeypatch):
    class EmptyResponse:
        text = "   "

    class EmptyModels:
        def generate_content(self, **kwargs):
            return EmptyResponse()

    class EmptyClient:
        models = EmptyModels()

    monkeypatch.setattr(answerer.genai, "Client", lambda api_key=None: EmptyClient())
    with pytest.raises(AnswerError):
        answer_question("q", HITS)


def test_prompt_contains_numbered_excerpts_and_the_question():
    prompt = build_prompt("what is the limit?", HITS)
    assert "[1] (a.md, part 0)" in prompt
    assert "[2] (b.md, part 2)" in prompt
    assert "Question: what is the limit?" in prompt
