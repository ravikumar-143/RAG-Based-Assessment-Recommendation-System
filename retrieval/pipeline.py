"""Full retrieval + rerank + balancing pipeline."""
from __future__ import annotations
from typing import List, Dict, Any

from retrieval.search import search_similar
from reranker.rerank import rerank_assessments
from utils import validate_recommendation_schema


def recommend(
    query: str,
    top_k: int = 30,
    max_results: int = 10,
    strict_mode: bool = False,
    enable_llm: bool = True,
) -> List[Dict[str, Any]]:
    candidates = search_similar(query, top_k=top_k)
    ranked = rerank_assessments(query, candidates, strict_mode=strict_mode, enable_llm=enable_llm)
    validated = validate_recommendation_schema(ranked[:max_results])
    return validated
