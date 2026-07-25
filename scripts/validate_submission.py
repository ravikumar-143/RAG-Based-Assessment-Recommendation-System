"""Validate submission CSV format and constraints."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SUBMISSION_PATH, TEST_DATA_PATH, METADATA_PATH  # noqa: E402
from utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def _load_metadata_urls() -> set[str]:
    urls = set()
    if METADATA_PATH.exists():
        import json

        data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        for item in data:
            url = item.get("url")
            if url:
                urls.add(url)
    return urls


def validate(submission_path: Path = SUBMISSION_PATH) -> None:
    if not submission_path.exists():
        raise SystemExit(f"Submission file not found: {submission_path}")

    df = pd.read_csv(submission_path)
    expected_cols = ["Query", "Assessment_url"]
    if list(df.columns) != expected_cols:
        raise SystemExit(f"Submission columns must be exactly {expected_cols}, got {list(df.columns)}")

    if df.empty:
        raise SystemExit("Submission CSV is empty")

    # No whitespace padding
    for col in expected_cols:
        stripped = df[col].astype(str).str.strip()
        if not stripped.equals(df[col].astype(str)):
            raise SystemExit(f"Column {col} contains leading/trailing whitespace")
        if stripped.eq("").any():
            raise SystemExit(f"Column {col} contains empty values")
        df[col] = stripped

    if df.isnull().any().any():
        raise SystemExit("Submission CSV contains blank cells")

    # Group-level checks
    for query, group in df.groupby("Query"):
        n = len(group)
        if n < 1:
            raise SystemExit(f"Query '{query}' has no recommendations")
        if n > 10:
            raise SystemExit(f"Query '{query}' has more than 10 recommendations ({n})")

    # URLs must exist in metadata
    metadata_urls = _load_metadata_urls()
    missing = [url for url in df["Assessment_url"] if url not in metadata_urls]
    if missing:
        raise SystemExit(f"Submission contains URLs not in metadata: {missing[:3]}{'...' if len(missing)>3 else ''}")

    # Query coverage must match test_dataset
    if not TEST_DATA_PATH.exists():
        raise SystemExit(f"test_dataset.csv not found at {TEST_DATA_PATH}")
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_queries = set(test_df["Query"].astype(str))
    submission_queries = set(df["Query"].astype(str))
    if submission_queries != test_queries:
        missing_queries = test_queries - submission_queries
        extra_queries = submission_queries - test_queries
        raise SystemExit(
            f"Submission queries mismatch test set. Missing={missing_queries or None}, Extra={extra_queries or None}"
        )

    logger.info("Final submission FULLY VALIDATED")
    print("Final submission FULLY VALIDATED")


if __name__ == "__main__":  # pragma: no cover
    validate()
