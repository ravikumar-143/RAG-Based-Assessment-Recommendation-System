"""Domain balance verification across training queries.

This script is intentionally light on Groq usage. By default it **does not**
call the LLM reranker (strict_mode=False) to avoid hitting rate limits while
generating the report. Use the ``--strict`` flag if you explicitly want to
exercise the Groq reranker during the check.
"""
from __future__ import annotations
import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRAIN_DATA_PATH  # noqa: E402
from groq import RateLimitError  # noqa: E402
from retrieval.pipeline import recommend  # noqa: E402
from reranker.rerank import TEST_TYPE_DOMAIN_MAP  # noqa: E402
from reranker.domain_detection import detect_domains  # noqa: E402
from utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def domains_for_item(item):
    return {TEST_TYPE_DOMAIN_MAP.get(t, "other") for t in item.get("test_type", [])}


def analyze_query(query: str, *, top_k: int = 8, strict_mode: bool = False, enable_llm: bool = False):
    try:
        results = recommend(query, top_k=top_k, max_results=10, strict_mode=strict_mode, enable_llm=enable_llm)
    except RateLimitError as exc:
        logger.warning("Rate limit during domain check for '%s'; falling back to non-strict rerank: %s", query, exc)
        results = recommend(query, top_k=top_k, max_results=10, strict_mode=False, enable_llm=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Strict rerank failed for '%s' (%s); retrying with non-strict fallback", query, exc)
        results = recommend(query, top_k=top_k, max_results=10, strict_mode=False, enable_llm=False)
    if not results:
        raise SystemExit(f"No recommendations returned for query: {query}; ensure index is built")

    test_types = []
    domain_counts = defaultdict(int)
    for item in results:
        test_types.extend(item.get("test_type", []))
        for d in domains_for_item(item):
            domain_counts[d] += 1

    test_type_dist = Counter(test_types)
    detected = detect_domains(query)
    active_domains = [d for d, c in detected.items() if c > 0]

    balanced = True
    if len(active_domains) >= 2:
        sorted_detected = sorted(detected.items(), key=lambda kv: kv[1], reverse=True)
        primary = sorted_detected[0][0]
        secondary = sorted_detected[1][0]
        threshold = max(1, math.ceil(0.3 * len(results)))
        secondary_count = domain_counts.get(secondary, 0)
        balanced = secondary_count >= threshold

    return {
        "query": query,
        "test_type_distribution": dict(test_type_dist),
        "domain_counts": dict(domain_counts),
        "detected_domains": detected,
        "balanced": balanced,
        "is_multi_domain": len([d for d, c in detected.items() if c > 0]) >= 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain balance verification across training queries")
    parser.add_argument("--strict", action="store_true", help="Use Groq reranker (may hit rate limits)")
    parser.add_argument("--top-k", type=int, default=6, dest="top_k", help="Number of candidates to fetch before rerank")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to pause between queries to reduce load")
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_DATA_PATH)
    if df.empty:
        raise SystemExit("train_dataset.csv is empty; populate it before running domain balance check")

    rows = []
    multi_domain = 0
    multi_domain_balanced = 0

    for query in df["Query"].unique():
        result = analyze_query(query, top_k=args.top_k, strict_mode=args.strict, enable_llm=args.strict)
        rows.append(result)
        if result["is_multi_domain"]:
            multi_domain += 1
            if result["balanced"]:
                multi_domain_balanced += 1
        if args.sleep:
            time.sleep(args.sleep)

    percent_balanced = (multi_domain_balanced / multi_domain * 100) if multi_domain else 100.0

    lines = []
    for r in rows:
        lines.append(f"Query: {r['query']}")
        lines.append(f"Detected domains: {r['detected_domains']}")
        lines.append(f"Domain counts (top 10): {r['domain_counts']}")
        lines.append(f"Test type distribution: {r['test_type_distribution']}")
        lines.append(f"Balanced: {r['balanced']}")
        lines.append("")

    lines.append(f"Multi-domain queries balanced: {percent_balanced:.2f}%")
    report_path = PROJECT_ROOT / "domain_balance_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("Domain balance report saved to", report_path)
    print(f"% of multi-domain queries successfully balanced: {percent_balanced:.2f}%")


if __name__ == "__main__":  # pragma: no cover
    main()
