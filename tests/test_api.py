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
            "sources": [
                {"source": h["source"], "position": h["position"]} for h in hits
            ],
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
    response = client.post(
        "/ask", data={"session_id": session_id, "question": "what is the upload limit"}
    )
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
    response = client.post(
        "/ask", data={"session_id": session_id, "question": "upload limit"}
    )
    assert response.status_code == 502
