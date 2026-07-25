"""FastAPI backend for SHL Assessment Recommendation Engine."""
from __future__ import annotations
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieval.pipeline import recommend
from utils import get_logger

load_dotenv()
logger = get_logger(__name__)

app = FastAPI(title="SHL Assessment Recommendation Engine", version="1.0")


class RecommendRequest(BaseModel):
    query: str = Field(..., description="User query")


class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[int]
    remote_support: str
    test_type: List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentResponse]


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_endpoint(payload: RecommendRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    try:
        results = recommend(query, top_k=30, max_results=10)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise HTTPException(status_code=500, detail="Index not built. Run embedding builder.")
    except ValueError as exc:
        logger.error(f"Schema validation failed: {exc}")
        raise HTTPException(status_code=500, detail="Invalid recommendation schema")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Recommendation failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal error")

    if not results:
        raise HTTPException(status_code=404, detail="No recommendations found")

    # enforce 1..10
    results = results[:10]
    return {"recommended_assessments": results}
