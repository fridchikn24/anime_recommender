
from .retriever import AnimeRetriever
from .reranker import LLmReranker

class AnimeRecommender:
    def __init__(self):
        self.retriever = AnimeRetriever(k=10)
        self.reranker = LLmReranker()

    def recommend(self, query: str, top_k=5):
        candidates = self.retriever.get_candidates(query)
        ranked = self.reranker.rerank(query, candidates)
        return ranked[:top_k]
