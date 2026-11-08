import os
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.answerer import NO_ANSWER, AnswerError, answer_question
from src.chunker import chunk_document
from src.embedding_retriever import EmbeddingError, EmbeddingRetriever
from src.hybrid_retriever import HybridRetriever
from src.loader import SUPPORTED, DocumentError, load_document
from src.retriever import Retriever

METHODS = {"tfidf", "embedding", "hybrid"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "20"))
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()
]

app = FastAPI(title="Ask My Docs API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = OrderedDict()


def _purge():
    now = time.monotonic()
    for key in [k for k, (t, _) in sessions.items() if now - t > SESSION_TTL_SECONDS]:
        del sessions[key]


def _get_session(session_id):
    _purge()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return sessions[session_id][1]


def build_retriever(chunks, method):
    if method == "tfidf":
        return Retriever(chunks)
    if method == "embedding":
        return EmbeddingRetriever(chunks)
    return HybridRetriever([Retriever(chunks), EmbeddingRetriever(chunks)])


@app.get("/health")
def health():
    _purge()
    return {"status": "ok", "sessions": len(sessions)}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...), method: str = Form("tfidf")):
    if method not in METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method. Use one of: {', '.join(sorted(METHODS))}",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Use one of: {', '.join(sorted(SUPPORTED))}",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. The maximum allowed size is {MAX_UPLOAD_MB} MB.",
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(file.filename).name
        path.write_bytes(data)
        try:
            document = load_document(path)
        except DocumentError as e:
            raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_document(document)
    try:
        retriever = build_retriever(chunks, method)
    except EmbeddingError as e:
        raise HTTPException(status_code=502, detail=str(e))

    session_id = str(uuid.uuid4())
    _purge()
    sessions[session_id] = (time.monotonic(), retriever)
    while len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)

    return {
        "session_id": session_id,
        "source": document["source"],
        "chunks": len(chunks),
        "method": method,
    }


@app.post("/ask")
def ask(session_id: str = Form(...), question: str = Form(...)):
    if not question.strip():
        raise HTTPException(status_code=400, detail="The question is empty")
    retriever = _get_session(session_id)
    try:
        hits = retriever.search(question, k=3)
        result = answer_question(question, hits)
    except (AnswerError, EmbeddingError) as e:
        raise HTTPException(status_code=502, detail=str(e))

    result["refused"] = result["answer"].strip() == NO_ANSWER
    result["passages"] = [
        {
            "source": h["source"],
            "position": h["position"],
            "text": h["text"],
            "score": round(h["score"], 3),
        }
        for h in hits
    ]
    return result
