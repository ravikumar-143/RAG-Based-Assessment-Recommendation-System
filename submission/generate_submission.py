"""Generate submission.csv from test_dataset.csv using recommendation pipeline and validate format."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TEST_DATA_PATH, SUBMISSION_PATH  # noqa: E402
from retrieval.pipeline import recommend  # noqa: E402
from scripts.validate_submission import validate  # noqa: E402
from utils import get_logger  # noqa: E402

logger = get_logger(__name__)
load_dotenv()


def generate_submission(top_k: int = 30, max_results: int = 10) -> None:
    if not TEST_DATA_PATH.exists():
        raise SystemExit(f"test_dataset.csv not found at {TEST_DATA_PATH}")

    df = pd.read_csv(TEST_DATA_PATH)
    expected_cols = ["Query"]
    if list(df.columns) != expected_cols:
        raise SystemExit(f"test_dataset.csv must have columns {expected_cols}, got {list(df.columns)}")
    if df.empty:
        raise SystemExit("test_dataset.csv is empty; populate it before generating submission")

    rows = []
    for _, row in df.iterrows():
        query = row["Query"]
        results = recommend(query, top_k=top_k, max_results=max_results, strict_mode=True)
        if not results:
            logger.warning("No recommendations for query: %s", query)
            continue
        for rec in results[:max_results]:
            rows.append({"Query": query, "Assessment_url": rec.get("url", "")})

    if not rows:
        raise SystemExit("No recommendations produced; cannot create submission")

    submission_df = pd.DataFrame(rows, columns=["Query", "Assessment_url"])
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Saved submission to %s", SUBMISSION_PATH)

    validate(SUBMISSION_PATH)


if __name__ == "__main__":  # pragma: no cover
    generate_submission()
