from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("chunks must not be empty")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in chunks)

    def search(self, question, k=3):
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [
            {
                "source": self.chunks[i]["source"],
                "position": self.chunks[i]["position"],
                "text": self.chunks[i]["text"],
                "score": float(scores[i]),
            }
            for i in order
            if scores[i] > 0
        ]
