RRF_CONSTANT = 60


class HybridRetriever:
    def __init__(self, retrievers, pool=10):
        if not retrievers:
            raise ValueError("retrievers must not be empty")
        self.retrievers = retrievers
        self.pool = pool

    def search(self, question, k=3):
        scores = {}
        chunks = {}
        for retriever in self.retrievers:
            results = retriever.search(question, k=self.pool)
            for rank, hit in enumerate(results, start=1):
                key = (hit["source"], hit["position"])
                chunks[key] = hit
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_CONSTANT + rank)

        ordered = sorted(scores, key=scores.get, reverse=True)[:k]
        return [
            {
                "source": chunks[key]["source"],
                "position": chunks[key]["position"],
                "text": chunks[key]["text"],
                "score": scores[key],
            }
            for key in ordered
        ]
