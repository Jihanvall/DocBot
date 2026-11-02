import argparse
import json

from src.chunker import chunk_document
from src.embedding_retriever import EmbeddingRetriever
from src.hybrid_retriever import HybridRetriever
from src.loader import load_document
from src.retriever import Retriever

parser = argparse.ArgumentParser()
parser.add_argument("--doc", default="data/readme.md")
parser.add_argument("--questions", default="eval/questions.json")
args = parser.parse_args()

with open(args.questions, encoding="utf-8") as f:
    questions = [q for q in json.load(f) if q["source"]]
chunks = chunk_document(load_document(args.doc), size=120, overlap=30)
print(len(chunks), "chunks,", len(questions), "answerable questions")

tfidf = Retriever(chunks)
embedding = EmbeddingRetriever(chunks)
methods = {
    "tf-idf": tfidf,
    "embedding": embedding,
    "hybrid": HybridRetriever([tfidf, embedding]),
}

for name, retriever in methods.items():
    for k in (1, 3):
        hits = 0
        for q in questions:
            results = retriever.search(q["question"], k=k)
            if any(all(w.lower() in h["text"].lower() for w in q["keywords"]) for h in results):
                hits += 1
        print(f"{name:<10} k={k}  {hits / len(questions):.2f}")
