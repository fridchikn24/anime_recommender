
import pandas as pd
from .store import ChromaStore
from .config import TEXT_COLUMN

class AnimeIngestor:
    def __init__(self, text_col=TEXT_COLUMN):
        self.text_col = text_col
        self.store = ChromaStore()

    def ingest_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        assert self.text_col in df.columns, f"{self.text_col} not in CSV!"

        self.store.from_dataframe(df, text_col=self.text_col)
        print("✓ Ingestion complete — embeddings saved to Chroma.")
