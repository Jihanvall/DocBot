import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = """You answer questions using only the numbered excerpts provided.
- If the excerpts do not contain the answer, reply exactly: I don't know based on the provided documents.
- Do not use outside knowledge and do not invent facts.
- Keep the answer short and cite the excerpts you used like [1] or [2]."""


class AnswerError(Exception):
    pass


def build_prompt(question, hits):
    parts = [f"[{i}] ({h['source']}, part {h['position']})\n{h['text']}" for i, h in enumerate(hits, 1)]
    return "Excerpts:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"


def answer_question(question, hits):
    if not hits:
        return {"answer": NO_ANSWER, "sources": []}

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=MODEL,
            contents=build_prompt(question, hits),
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        text = (response.text or "").strip()
    except Exception as e:
        raise AnswerError(f"Could not get an answer from Gemini: {e}")

    if not text:
        raise AnswerError("Gemini returned an empty answer")

    sources = [{"source": h["source"], "position": h["position"]} for h in hits]
    return {"answer": text, "sources": sources}
