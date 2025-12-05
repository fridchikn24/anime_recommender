
from fastapi import FastAPI
from pydantic import BaseModel
from src.recommender.engine import AnimeRecommender

app = FastAPI(
    title="Anime RAG Recommender API",
    version="1.0.0"
)

# Load engine once (cached)
engine = AnimeRecommender()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/recommend")
def recommend(req: QueryRequest):
    results = engine.recommend(req.query, req.top_k)
    return {"results": [r.page_content for r in results]}
