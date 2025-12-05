
from .store import ChromaStore

class AnimeRetriever:
    def __init__(self, k=5):
        self.store = ChromaStore()
        self.store.load()
        self.retriever = self.store.as_retriever(k=k)

    def get_candidates(self, query: str):
        return self.retriever.invoke(query)
