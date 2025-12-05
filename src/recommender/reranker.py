
from langchain_openai import ChatOpenAI
from .config import LLM_MODEL

class LLmReranker:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    def rerank(self, query: str, docs):
        doc_strings = [
            f"Title: {doc.metadata.get('title')}\nSynopsis: {doc.page_content}"
            for doc in docs
        ]

        prompt = f"""
            You are a recommendation engine for anime.
            User search: {query}

            Below are candidate anime:
            {chr(10).join(f"[{i}] {d}" for i, d in enumerate(doc_strings))}

            Rank them from best to worst match. Return only their indices in order.
            """

        response = self.llm.invoke(prompt)
        order = [int(x) for x in response.content.split() if x.isdigit()]

        return [docs[i] for i in order]
