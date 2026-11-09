import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(autouse=True)
def fake_gemini(monkeypatch):
    monkeypatch.setattr(
        api,
        "answer_question",
        lambda question, hits: {
            "answer": "20 MB [1]",
            "sources": [{"source": h["source"], "position": h["position"]} for h in hits],
        },
    )
    api.sessions.clear()


@pytest.fixture
def client():
    return TestClient(api.app)


def _upload(client, name="notes.txt", content=b"the upload limit is 20 megabytes"):
    return client.post("/documents", files={"file": (name, content, "text/plain")})


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_upload_returns_a_session(client):
    body = _upload(client).json()
    assert body["source"] == "notes.txt"
    assert body["chunks"] >= 1
    assert "session_id" in body


def test_unsupported_file_is_rejected(client):
    assert _upload(client, "photo.png", b"x").status_code == 400


def test_empty_file_is_rejected(client):
    assert _upload(client, "empty.txt", b"  ").status_code == 400


def test_oversized_file_is_rejected(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_MB", 0)
    assert _upload(client).status_code == 413


def test_ask_returns_answer_and_sources(client):
    session_id = _upload(client).json()["session_id"]
    response = client.post("/ask", data={"session_id": session_id, "question": "what is the upload limit"})
    assert response.status_code == 200
    assert response.json()["answer"] == "20 MB [1]"
    assert response.json()["sources"][0]["source"] == "notes.txt"


def test_ask_unknown_session_returns_404(client):
    response = client.post("/ask", data={"session_id": "nope", "question": "q"})
    assert response.status_code == 404


def test_empty_question_returns_400(client):
    session_id = _upload(client).json()["session_id"]
    response = client.post("/ask", data={"session_id": session_id, "question": "  "})
    assert response.status_code == 400


def test_expired_session_returns_404(client, monkeypatch):
    session_id = _upload(client).json()["session_id"]
    monkeypatch.setattr(api, "SESSION_TTL_SECONDS", -1)
    response = client.post("/ask", data={"session_id": session_id, "question": "q"})
    assert response.status_code == 404


def test_gemini_failure_returns_502(client, monkeypatch):
    def fail(question, hits):
        raise api.AnswerError("boom")

    monkeypatch.setattr(api, "answer_question", fail)
    session_id = _upload(client).json()["session_id"]
    response = client.post("/ask", data={"session_id": session_id, "question": "upload limit"})
    assert response.status_code == 502


class FakeEmbeddingRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, question, k=3):
        return [{**c, "score": 0.9} for c in self.chunks[:k]]


def test_ask_returns_the_retrieved_passages(client):
    session_id = _upload(client).json()["session_id"]
    body = client.post("/ask", data={"session_id": session_id, "question": "what is the upload limit"}).json()
    assert body["refused"] is False
    passage = body["passages"][0]
    assert passage["source"] == "notes.txt"
    assert "20 megabytes" in passage["text"]


def test_refusal_is_flagged(client, monkeypatch):
    monkeypatch.setattr(api, "answer_question", lambda q, h: {"answer": api.NO_ANSWER, "sources": []})
    session_id = _upload(client).json()["session_id"]
    body = client.post("/ask", data={"session_id": session_id, "question": "upload limit"}).json()
    assert body["refused"] is True


def test_unknown_method_is_rejected(client):
    response = client.post(
        "/documents",
        files={"file": ("a.txt", b"hello world", "text/plain")},
        data={"method": "magic"},
    )
    assert response.status_code == 400


def test_embedding_method_is_used_when_chosen(client, monkeypatch):
    monkeypatch.setattr(api, "EmbeddingRetriever", FakeEmbeddingRetriever)
    response = client.post(
        "/documents",
        files={"file": ("a.txt", b"the upload limit is 20 megabytes", "text/plain")},
        data={"method": "embedding"},
    )
    assert response.json()["method"] == "embedding"
    session_id = response.json()["session_id"]
    body = client.post("/ask", data={"session_id": session_id, "question": "limit"}).json()
    assert body["passages"][0]["score"] == 0.9


def test_embedding_failure_on_upload_returns_502(client, monkeypatch):
    class Failing:
        def __init__(self, chunks):
            raise api.EmbeddingError("quota")

    monkeypatch.setattr(api, "EmbeddingRetriever", Failing)
    response = client.post(
        "/documents",
        files={"file": ("a.txt", b"hello world", "text/plain")},
        data={"method": "embedding"},
    )
    assert response.status_code == 502


def test_follow_up_uses_the_rewritten_question(client, monkeypatch):
    monkeypatch.setattr(api, "rewrite_question", lambda q, h: "upload limit megabytes")
    session_id = _upload(client).json()["session_id"]
    history = '[{"role": "user", "text": "what is the upload limit?"}]'
    body = client.post(
        "/ask", data={"session_id": session_id, "question": "and in total?", "history": history}
    ).json()
    assert body["standalone_question"] == "upload limit megabytes"


def test_without_history_the_question_is_not_rewritten(client, monkeypatch):
    seen = []
    monkeypatch.setattr(api, "rewrite_question", lambda q, h: seen.append(h) or q)
    session_id = _upload(client).json()["session_id"]
    body = client.post("/ask", data={"session_id": session_id, "question": "upload limit"}).json()
    assert seen == [[]]
    assert body["standalone_question"] == "upload limit"


def test_invalid_history_is_rejected(client):
    session_id = _upload(client).json()["session_id"]
    response = client.post("/ask", data={"session_id": session_id, "question": "q", "history": "not json"})
    assert response.status_code == 400
