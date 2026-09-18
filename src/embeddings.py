from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the SentenceTransformer embedding model."""
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts, model=None):
    """
    Generate embeddings for a list of text units.

    Args:
        texts: List of strings.
        model: Optional preloaded SentenceTransformer model.

    Returns:
        NumPy array of embeddings.
    """
    if model is None:
        model = load_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


if __name__ == "__main__":
    sample_texts = [
        "Prompt compression reduces the number of tokens.",
        "Compression should preserve important information.",
        "Graph-based methods represent semantic relationships."
    ]

    model = load_embedding_model()
    embeddings = generate_embeddings(sample_texts, model)

    print("Embeddings generated successfully")
    print("Number of text units:", len(sample_texts))
    print("Embedding shape:", embeddings.shape)
    print("First embedding:", embeddings[0][:5])