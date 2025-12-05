import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from .config import CHROMA_DB_DIR, EMBED_MODEL



class ChromaStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        self.db = None
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    def from_dataframe(self, df: pd.DataFrame, text_col: str):
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame")

        docs = df[text_col].astype(str).tolist()
        metadatas = df.to_dict(orient="records")

        self.db = Chroma.from_texts(
            texts=docs,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=CHROMA_DB_DIR
        )
        self.db.persist()
        print(f"✓ Chroma database created at '{CHROMA_DB_DIR}' with {len(docs)} documents")

    def load(self):
        if not os.path.exists(CHROMA_DB_DIR):
            raise FileNotFoundError(f"Chroma DB not found at {CHROMA_DB_DIR}. Run ingestion first.")
        self.db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print(f"✓ Chroma database loaded from '{CHROMA_DB_DIR}'")
        return self.db

    def as_retriever(self, k=5):
        if not self.db:
            raise ValueError("Chroma DB not loaded. Call load() first.")
        return self.db.as_retriever(search_kwargs={"k": k})
