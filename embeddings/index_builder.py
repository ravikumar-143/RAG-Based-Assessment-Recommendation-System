import os
import pickle

import faiss
import pandas as pd

from embeddings.embedder import generate_embeddings

DATA_PATH = "data/cleaned_catalog.csv"

FAISS_DIR = "data/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")


def build_index():
    df = pd.read_csv(DATA_PATH)

    # Combine all columns into one searchable text
    texts = df.astype(str).agg(" ".join, axis=1).tolist()

    embeddings = generate_embeddings(texts)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(FAISS_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(df.to_dict("records"), f)

    print("FAISS index built successfully.")
    print(f"Indexed {len(df)} assessments.")


def load_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "Index not found. Run index_builder.py first."
        )

    index = faiss.read_index(INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


if __name__ == "__main__":
    build_index()