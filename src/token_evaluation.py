import tiktoken
from pathlib import Path


# --------------------------------------------------
# File paths
# --------------------------------------------------

ORIGINAL_PROMPT_PATH = Path(
    "data/prompts/sample_prompt.txt"
)

COMPRESSED_PROMPT_PATH = Path(
    "data/compressed_prompt.txt"
)


# --------------------------------------------------
# Load prompt
# --------------------------------------------------

def load_prompt(path):
    """Load prompt text from a file."""

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# --------------------------------------------------
# Count tokens
# --------------------------------------------------

def count_tokens(text):
    """
    Count tokens using the cl100k_base tokenizer.

    This tokenizer is commonly used for OpenAI-style
    language models and provides a consistent basis
    for comparing the original and compressed prompts.
    """

    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )

    tokens = encoding.encode(text)

    return len(tokens)


# --------------------------------------------------
# Calculate compression metrics
# --------------------------------------------------

def calculate_metrics(
    original_tokens,
    compressed_tokens
):
    """Calculate token compression metrics."""

    tokens_reduced = (
        original_tokens -
        compressed_tokens
    )

    if original_tokens == 0:
        token_reduction = 0.0
        compression_ratio = 0.0
    else:
        token_reduction = (
            tokens_reduced /
            original_tokens
        )

        compression_ratio = (
            original_tokens /
            compressed_tokens
            if compressed_tokens > 0
            else 0.0
        )

    return (
        tokens_reduced,
        token_reduction,
        compression_ratio
    )


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    print("Loading prompts...")

    original_prompt = load_prompt(
        ORIGINAL_PROMPT_PATH
    )

    compressed_prompt = load_prompt(
        COMPRESSED_PROMPT_PATH
    )

    print("Counting tokens...")

    original_tokens = count_tokens(
        original_prompt
    )

    compressed_tokens = count_tokens(
        compressed_prompt
    )

    (
        tokens_reduced,
        token_reduction,
        compression_ratio
    ) = calculate_metrics(
        original_tokens,
        compressed_tokens
    )

    print("\n===================================")
    print("TOKEN COMPRESSION EVALUATION")
    print("===================================")

    print(
        f"Original tokens: "
        f"{original_tokens}"
    )

    print(
        f"Compressed tokens: "
        f"{compressed_tokens}"
    )

    print(
        f"Tokens reduced: "
        f"{tokens_reduced}"
    )

    print(
        f"Token reduction: "
        f"{token_reduction * 100:.2f}%"
    )

    print(
        f"Compression ratio: "
        f"{compression_ratio:.2f}x"
    )


# --------------------------------------------------
# Run program
# --------------------------------------------------

if __name__ == "__main__":
    main()