"""Validate training dataset against catalog metadata."""
from __future__ import annotations
import sys
import json
import pandas as pd
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRAIN_DATA_PATH, METADATA_PATH  # noqa: E402
from utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    if not TRAIN_DATA_PATH.exists():
        raise SystemExit(f"train_dataset.csv missing at {TRAIN_DATA_PATH}")
    if not METADATA_PATH.exists():
        raise SystemExit(f"metadata.json missing at {METADATA_PATH}; build index first")

    df = pd.read_csv(TRAIN_DATA_PATH)
    expected_cols = ["Query", "Assessment_url"]
    if list(df.columns) != expected_cols:
        raise SystemExit(f"Columns must be exactly {expected_cols}, got {list(df.columns)}")
    if df.empty:
        raise SystemExit("train_dataset.csv is empty; add training rows")

    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    catalog_urls = {item["url"] for item in meta}

    errors = []

    # Blank row check
    if df.isnull().any().any():
        errors.append("CSV contains blank cells")

    # URL existence and duplicates per query
    for query, group in df.groupby("Query"):
        urls = group["Assessment_url"].tolist()
        # duplicates per query
        if len(urls) != len(set(urls)):
            errors.append(f"Duplicate URLs found for query: {query}")
        for url in urls:
            if url not in catalog_urls:
                errors.append(f"URL not in metadata: {url}")

    if errors:
        msg = "\n".join(errors)
        raise SystemExit(f"Train data validation failed:\n{msg}")

    logger.info("Train dataset validation passed: %d rows, %d queries", len(df), df['Query'].nunique())
    print("Train dataset validation passed")


if __name__ == "__main__":  # pragma: no cover
    main()
