import os
from dotenv import load_dotenv

load_dotenv()

# Column in CSV containing text to embed
TEXT_COLUMN = "synopsis"

# Directory to store Chroma database
CHROMA_DB_DIR = "chroma_db"

# OpenAI embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# OpenAI API key (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")