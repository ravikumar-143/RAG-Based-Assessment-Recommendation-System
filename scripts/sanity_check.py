"""Lightweight sanity checks without calling external LLMs."""
from __future__ import annotations
import json
from pathlib import Path
import sys

# Ensure project root on path when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLEANED_CATALOG_PATH, FAISS_INDEX_PATH, METADATA_PATH
from retrieval.search import search_similar


def main():
    # Check catalog size
    if not CLEANED_CATALOG_PATH.exists():
        raise SystemExit("cleaned_catalog.json missing; run scraper first")
    data = json.loads(CLEANED_CATALOG_PATH.read_text(encoding="utf-8"))
    count = len(data)
    if count < 377:
        raise SystemExit(f"Catalog has {count} items (<377). Scrape again.")
    print(f"Catalog OK: {count} items")

    # Check index and metadata
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise SystemExit("FAISS index/metadata missing; run embeddings/index_builder.py")
    print("Index and metadata files present")

    # Run a sample retrieval (no Groq call)
    sample_query = "data analyst python sql"
    results = search_similar(sample_query, top_k=3)
    if not results:
        raise SystemExit("Search returned no results; check index build")
    print("Sample retrieval top 3:")
    for r in results:
        print(f"- {r.get('name')} ({r.get('url')})")


if __name__ == "__main__":  # pragma: no cover
    main()
