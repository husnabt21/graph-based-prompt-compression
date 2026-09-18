from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


ORIGINAL_PROMPT_PATH = Path(
    "data/prompts/sample_prompt.txt"
)

COMPRESSED_PROMPT_PATH = Path(
    "data/compressed_prompt.txt"
)

MODEL_NAME = "all-MiniLM-L6-v2"


def load_text(path):
    """Load text from a file."""

    return path.read_text(
        encoding="utf-8"
    ).strip()


def generate_embedding(text, model):
    """Generate a normalized sentence embedding."""

    embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding


def calculate_semantic_similarity(
    original_embedding,
    compressed_embedding
):
    """Calculate cosine similarity between prompts."""

    similarity = cosine_similarity(
        original_embedding,
        compressed_embedding
    )

    return float(similarity[0][0])


def main():

    print("Loading prompts...")

    original_prompt = load_text(
        ORIGINAL_PROMPT_PATH
    )

    compressed_prompt = load_text(
        COMPRESSED_PROMPT_PATH
    )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Generating original prompt embedding...")

    original_embedding = generate_embedding(
        original_prompt,
        model
    )

    print("Generating compressed prompt embedding...")

    compressed_embedding = generate_embedding(
        compressed_prompt,
        model
    )

    print("\nCalculating semantic similarity...")

    similarity = calculate_semantic_similarity(
        original_embedding,
        compressed_embedding
    )

    print("\n===================================")
    print("SEMANTIC PRESERVATION EVALUATION")
    print("===================================")

    print(
        f"Semantic similarity: {similarity:.4f}"
    )

    print(
        f"Semantic similarity (%): "
        f"{similarity * 100:.2f}%"
    )


if __name__ == "__main__":
    main()