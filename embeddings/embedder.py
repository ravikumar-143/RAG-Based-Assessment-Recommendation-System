from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def generate_embeddings(texts):
    """
    Generate embeddings for a list of text strings.
    """
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return embeddings