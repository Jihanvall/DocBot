import argparse
import json
import time

from src.answerer import answer_question
from src.chunker import chunk_document
from src.embedding_retriever import EmbeddingRetriever
from src.evaluation import contains_keywords, hit_at_k, is_refusal, summarize
from src.hybrid_retriever import HybridRetriever
from src.loader import load_document
from src.retriever import Retriever

parser = argparse.ArgumentParser()
parser.add_argument("--doc", default="data/readme.md")
parser.add_argument("--questions", default="eval/questions.json")
parser.add_argument("--last", type=int, default=None)
parser.add_argument("--method", choices=["tfidf", "embedding", "hybrid"], default="tfidf")
args = parser.parse_args()

K = 3
PAUSE_SECONDS = 5

with open(args.questions, encoding="utf-8") as f:
    questions = json.load(f)
if args.last:
    questions = questions[-args.last :]

chunks = chunk_document(load_document(args.doc), size=120, overlap=30)
if args.method == "tfidf":
    retriever = Retriever(chunks)
elif args.method == "embedding":
    retriever = EmbeddingRetriever(chunks)
else:
    retriever = HybridRetriever([Retriever(chunks), EmbeddingRetriever(chunks)])

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
    print("      answer:", answer[:300].replace("\n", " "))

print(json.dumps(summarize(rows), indent=2))
