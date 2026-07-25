"""Global configuration for SHL Assessment Recommendation Engine."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
EMBEDDING_DIR = BASE_DIR / "embeddings"
MODEL_CACHE_DIR = EMBEDDING_DIR / "model_cache"
SUBMISSION_DIR = BASE_DIR / "submission"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = EMBEDDING_DIR / "faiss.index"
METADATA_PATH = EMBEDDING_DIR / "metadata.json"
CLEANED_CATALOG_PATH = DATA_DIR / "cleaned_catalog.json"
TRAIN_DATA_PATH = DATA_DIR / "train_dataset.csv"
TEST_DATA_PATH = DATA_DIR / "test_dataset.csv"
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Ensure directories exist
for path in [DATA_DIR, LOG_DIR, EMBEDDING_DIR, SUBMISSION_DIR]:
    path.mkdir(parents=True, exist_ok=True)
