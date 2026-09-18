from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_RESPONSE_PATH = (
    BASE_DIR / "data" / "results" / "original_response.txt"
)

COMPRESSED_RESPONSE_PATH = (
    BASE_DIR / "data" / "results" / "compressed_response.txt"
)


def load_response(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def word_count(text):
    return len(text.split())


def keyword_set(text):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    stopwords = {
        "this", "that", "with", "from", "have",
        "which", "their", "there", "about",
        "these", "those", "should", "would",
        "could", "while", "where", "into",
        "also", "than", "then", "such",
        "associated", "following"
    }

    return {
        word for word in words
        if word not in stopwords
    }


def keyword_overlap(original, compressed):
    original_words = keyword_set(original)
    compressed_words = keyword_set(compressed)

    if not original_words:
        return 0.0

    overlap = original_words.intersection(compressed_words)

    return len(overlap) / len(original_words)


def main():

    print("=" * 60)
    print("RESPONSE QUALITY EVALUATION")
    print("=" * 60)

    original = load_response(
        ORIGINAL_RESPONSE_PATH
    )

    compressed = load_response(
        COMPRESSED_RESPONSE_PATH
    )

    original_count = word_count(original)
    compressed_count = word_count(compressed)

    overlap = keyword_overlap(
        original,
        compressed
    )

    print(f"\nOriginal response words: {original_count}")
    print(f"Compressed response words: {compressed_count}")

    print(
        f"\nResponse keyword overlap: "
        f"{overlap:.4f}"
    )

    print(
        f"Response keyword overlap (%): "
        f"{overlap * 100:.2f}%"
    )

    print("\n" + "=" * 60)
    print("RESPONSE QUALITY EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()