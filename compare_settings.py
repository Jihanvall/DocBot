import json

from src.chunker import chunk_document
from src.loader import load_document
from src.retriever import Retriever

questions = [
    q for q in json.load(open("eval/questions.json", encoding="utf-8")) if q["source"]
]
document = load_document("data/readme.md")

print("size overlap k hit_rate")
for size in (60, 120, 300):
    chunks = chunk_document(document, size=size, overlap=size // 4)
    retriever = Retriever(chunks)
    for k in (1, 3, 5):
        hits = sum(
            any(
                all(w.lower() in h["text"].lower() for w in q["keywords"])
                for h in retriever.search(q["question"], k=k)
            )
            for q in questions
        )
        print(f"{size:<5} {size // 4:<8} {k} {hits / len(questions):.2f}")
