"""Vector search using FAISS over SHL assessments."""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np

from embeddings.index_builder import load_index
from embeddings.embedder import generate_embeddings
from utils import get_logger

logger = get_logger(__name__)

# Query expansion to improve retrieval


QUERY_EXPANSIONS = {
    "azure data engineer":
        "azure data engineer azure synapse azure data factory adf databricks pyspark spark sql data warehouse etl pipelines",

    "python developer":
        "python developer django flask fastapi sql api backend programming",

    "react developer":
        "react javascript html css frontend node express redux",

    "machine learning engineer":
        "python machine learning deep learning tensorflow pytorch sklearn ai data science",

    "devops engineer":
        "docker kubernetes jenkins ci cd linux terraform cloud azure aws",

    "business analyst":
        "business analyst requirements sql excel communication stakeholder agile"
}


def search_similar(query: str, top_k: int = 50) -> List[Dict[str, Any]]:
    """
    Search the FAISS index for assessments similar to the user's query.
    """

    index, metadata = load_index()

    expanded_query = QUERY_EXPANSIONS.get(
        query.lower(),
        query
    )

    logger.info(f"Expanded query: {expanded_query}")

    query_embedding = generate_embeddings(
        [expanded_query]
    ).astype(np.float32)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        item = metadata[idx].copy()
        item["score"] = float(distance)
        results.append(item)

    logger.info(f"Retrieved {len(results)} similar assessments.")

    # Remove duplicate assessments
    unique = []
    seen = set()

    for item in results:
        key = item["url"]

        if key not in seen:
            seen.add(key)
            unique.append(item)

    logger.info(f"Retrieved {len(unique)} unique assessments.")

    return unique