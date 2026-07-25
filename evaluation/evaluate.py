"""Evaluation module computing Recall@10 on train dataset."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd
from dotenv import load_dotenv

# Ensure project root on path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure environment variables (e.g., GROQ_API_KEY) are loaded
load_dotenv()

from config import TRAIN_DATA_PATH
from retrieval.search import search_similar
from reranker.rerank import rerank_assessments, _balance_domains
from retrieval.pipeline import recommend
from utils import get_logger

logger = get_logger(__name__)


def recall_at_k(recommended: List[str], relevant: str, k: int = 10) -> float:
    return 1.0 if relevant in recommended[:k] else 0.0


def evaluate() -> None:
    df = pd.read_csv(TRAIN_DATA_PATH)
    if df.empty:
        raise ValueError("train_dataset.csv is empty; populate it before evaluation")

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set. Groq reranking must be active for evaluation.")

    per_query = []
    baseline_scores = []
    rerank_scores = []
    balanced_scores = []

    print("Groq reranking ACTIVE")

    for _, row in df.iterrows():
        query = row["Query"]
        truth_url = row["Assessment_url"]

        # Baseline: FAISS only
        baseline = search_similar(query, top_k=10)
        baseline_urls = [b.get("url", "") for b in baseline]
        baseline_recall = recall_at_k(baseline_urls, truth_url)
        baseline_scores.append(baseline_recall)

        # Rerank without balancing (strict Groq)
        reranked = rerank_assessments(
            query,
            search_similar(query, top_k=15),
            apply_balance=False,
            strict_mode=True,
        )
        rerank_urls = [r.get("url", "") for r in reranked]
        rerank_score = recall_at_k(rerank_urls, truth_url)
        rerank_scores.append(rerank_score)

        # After balancing: reuse reranked list to avoid a second Groq call (saves tokens)
        balanced = _balance_domains(query, reranked)
        balanced_urls = [b.get("url", "") for b in balanced]
        balanced_score = recall_at_k(balanced_urls, truth_url)
        balanced_scores.append(balanced_score)

        per_query.append(
            {
                "query": query,
                "truth": truth_url,
                "baseline_recall@10": baseline_recall,
                "rerank_recall@10": rerank_score,
                "balanced_recall@10": balanced_score,
            }
        )
        print("-" * 36)
        print(f"Query: {query}")
        print(f"Baseline Recall@10: {baseline_recall:.2f}")
        print(f"Reranked Recall@10: {rerank_score:.2f}")
        print(f"Balanced Recall@10: {balanced_score:.2f}")

    mean_baseline = sum(baseline_scores) / len(baseline_scores)
    mean_rerank = sum(rerank_scores) / len(rerank_scores)
    mean_balanced = sum(balanced_scores) / len(balanced_scores)

    logger.info("Per-query Recall@10:")
    for item in per_query:
        logger.info(item)

    logger.info("\nMean Recall@10:")
    logger.info(f"Baseline: {mean_baseline:.3f}")
    logger.info(f"Rerank: {mean_rerank:.3f}")
    logger.info(f"Balanced: {mean_balanced:.3f}")

    print("-" * 36)
    print("\nFinal Mean Recall@10:")
    print(f"Baseline: {mean_baseline:.3f}")
    print(f"Reranked: {mean_rerank:.3f}")
    print(f"Balanced: {mean_balanced:.3f}")

    # Summary table
    summary_df = pd.DataFrame(per_query)[
        ["query", "truth", "baseline_recall@10", "rerank_recall@10", "balanced_recall@10"]
    ]
    table_text = summary_df.to_string(index=False)
    mean_text = (
        f"Baseline Mean Recall@10: {mean_baseline:.3f}\n"
        f"Reranked Mean Recall@10: {mean_rerank:.3f}\n"
        f"Balanced Mean Recall@10: {mean_balanced:.3f}\n"
    )

    output_path = PROJECT_ROOT / "evaluation_results.txt"
    output_path.write_text(
        "Groq reranking ACTIVE\n\n" + table_text + "\n\n" + mean_text,
        encoding="utf-8",
    )
    logger.info("Saved evaluation results to %s", output_path)
    print("\nSummary Table:\n")
    print(table_text)
    print("\nResults saved to", output_path)


if __name__ == "__main__":  # pragma: no cover
    evaluate()
