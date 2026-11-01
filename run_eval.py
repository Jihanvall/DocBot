import json
import sys
import time

from src.answerer import answer_question
from src.chunker import chunk_document
from src.evaluation import contains_keywords, hit_at_k, is_refusal, summarize
from src.loader import load_document
from src.retriever import Retriever

DOC = "data/readme.md"
QUESTIONS = "eval/questions.json"
K = 3
PAUSE_SECONDS = 5

questions = json.load(open(QUESTIONS, encoding="utf-8"))
if len(sys.argv) > 1:
    questions = questions[-int(sys.argv[1]) :]
chunks = chunk_document(load_document(DOC), size=120, overlap=30)
retriever = Retriever(chunks)

rows = []
for item in questions:
    hits = retriever.search(item["question"], k=K)
    try:
        answer = answer_question(item["question"], hits)["answer"] if hits else None
    except Exception as e:
        print("STOPPED:", e)
        break
    if hits:
        time.sleep(PAUSE_SECONDS)
    answer = answer or "I don't know based on the provided documents."
    expected = item["source"]
    rows.append(
        {
            "question": item["question"],
            "expected_source": expected,
            "hit": hit_at_k(hits, expected) if expected else False,
            "correct": contains_keywords(answer, item["keywords"]) if expected else False,
            "refused": is_refusal(answer),
        }
    )
    row = rows[-1]
    status = "OK" if row["correct"] else ("REFUSED" if row["refused"] else "MISS")
    print(status.ljust(8), item["question"])
    if status == "MISS":
        print("         answer:", answer[:200])

print(json.dumps(summarize(rows), indent=2))
