import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
BATCH = 100


class EmbeddingError(Exception):
    pass


def embed_texts(texts, task_type=None):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    vectors = []
    try:
        for i in range(0, len(texts), BATCH):
            config = (
                types.EmbedContentConfig(task_type=task_type) if task_type else None
            )
            response = client.models.embed_content(
                model=EMBED_MODEL, contents=texts[i : i + BATCH], config=config
            )
            vectors.extend(e.values for e in response.embeddings)
    except Exception as e:
        raise EmbeddingError(f"Could not get embeddings from Gemini: {e}")
    return np.array(vectors, dtype=float)


class EmbeddingRetriever:
    def __init__(self, chunks, embed=embed_texts):
        if not chunks:
            raise ValueError("chunks must not be empty")
        self.chunks = chunks
        self.embed = embed
        self.matrix = self._normalize(
            embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
        )

    @staticmethod
    def _normalize(m):
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return m / norms

    def search(self, question, k=3):
        query = self._normalize(self.embed([question], "RETRIEVAL_QUERY"))
        scores = (self.matrix @ query[0]).ravel()
        order = scores.argsort()[::-1][:k]
        return [
            {
                "source": self.chunks[i]["source"],
                "position": self.chunks[i]["position"],
                "text": self.chunks[i]["text"],
                "score": float(scores[i]),
            }
            for i in order
        ]
