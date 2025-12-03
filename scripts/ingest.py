import sys
import os

# Add project root to module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.recommender.store import ChromaStore
from src.recommender.config import TEXT_COLUMN

CSV_PATH = "data/anime_dataset.csv"

def ingest_csv(csv_path=CSV_PATH, text_col=TEXT_COLUMN):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at '{csv_path}'")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV file is empty")

    store = ChromaStore()
    store.from_dataframe(df, text_col=text_col)
    print("✓ Ingestion complete")

if __name__ == "__main__":
    ingest_csv()
