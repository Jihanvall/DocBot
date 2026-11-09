import os

from dotenv import load_dotenv
from google import genai

from src.answerer import MODEL

load_dotenv()

MAX_TURNS = 6
MAX_QUESTION_CHARS = 300

REWRITE_INSTRUCTIONS = """Rewrite the last question so that it can be understood without the conversation.
- Use the conversation only to resolve references such as "it", "that" or "what happened next".
- If the question is already standalone, return it unchanged.
- Reply with the rewritten question only, with no explanation."""


def build_rewrite_prompt(question, history):
    lines = [REWRITE_INSTRUCTIONS, "", "Conversation:"]
    for turn in history[-MAX_TURNS:]:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['text']}")
    lines += ["", f"Last question: {question}", "", "Standalone question:"]
    return "\n".join(lines)


def rewrite_question(question, history):
    if not history:
        return question
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=MODEL, contents=build_rewrite_prompt(question, history)
        )
        text = (response.text or "").strip()
    except Exception:
        return question
    first_line = text.splitlines()[0].strip() if text else ""
    if not first_line or len(first_line) > MAX_QUESTION_CHARS:
        return question
    return first_line
