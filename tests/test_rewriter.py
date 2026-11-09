from src import rewriter
from src.rewriter import build_rewrite_prompt, rewrite_question

HISTORY = [
    {"role": "user", "text": "What changed human diet?"},
    {"role": "assisstant", "text": "Fire changed it [1]."},
]


class FakeResponse:
    def __init__(self, text):
        self.text = text


def install_client(monkeypatch, text=None, error=None):
    class FakeModels:
        def generate_content(self, **kwargs):
            if error:
                raise error
            return FakeResponse(text)

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(rewriter.genai, "Client", lambda api_key=None: FakeClient())


def test_no_history_returns_the_question_without_calling_gemini(monkeypatch):
    def boom(api_key=None):
        raise AssertionError("Gemini must not be called")

    monkeypatch.setattr(rewriter.genai, "Client", boom)
    assert rewrite_question("and then?", []) == "and then?"


def test_follow_up_is_rewritten(monkeypatch):
    install_client(monkeypatch, text="  what happened after fire changed the diet? ")
    assert rewrite_question("and then?", HISTORY) == "what happened after fire changed the diet?"


def test_gemini_failure_falls_back_to_the_original_question(monkeypatch):
    install_client(monkeypatch, error=RuntimeError("no network"))
    assert rewrite_question("and then?", HISTORY) == "and then?"


def test_empty_answer_falls_back_to_the_original_question(monkeypatch):
    install_client(monkeypatch, text="  ")
    assert rewrite_question("and then?", HISTORY) == "and then?"


def test_prompt_contains_the_conversation_and_the_question():
    prompt = build_rewrite_prompt("and then?", HISTORY)
    assert "User: What changed human diet?" in prompt
    assert "Assistant: Fire changed it [1]." in prompt
    assert "Last question: and then?" in prompt


def test_too_long_answer_falls_back_to_the_original_question(monkeypatch):
    install_client(monkeypatch, text="x" * (rewriter.MAX_QUESTION_CHARS + 1))
    assert rewrite_question("and then?", HISTORY) == "and then?"


def test_answer_at_the_limit_is_accepted(monkeypatch):
    text = "x" * rewriter.MAX_QUESTION_CHARS
    install_client(monkeypatch, text=text)
    assert rewrite_question("and then?", HISTORY) == text


def test_only_the_first_line_of_the_answer_is_used(monkeypatch):
    install_client(
        monkeypatch,
        text="What happened after fire changed the diet?\nI replaced 'that' with fire.",
    )
    assert rewrite_question("and then?", HISTORY) == "What happened after fire changed the diet?"
