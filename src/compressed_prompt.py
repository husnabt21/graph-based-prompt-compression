import json
from pathlib import Path


INPUT_PATH = Path("data/compressed_information_units.json")
OUTPUT_PATH = Path("data/compressed_prompt.txt")


def load_compressed_units(path):
    """Load compressed information units from JSON."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def reconstruct_prompt(units):
    """
    Reconstruct the compressed prompt while preserving
    the original information-unit order.
    """

    sentences = []

    for unit in units:
        text = unit["text"].strip()

        if text:
            sentences.append(text)

    return "\n".join(sentences)


def save_compressed_prompt(prompt, path):
    """Save reconstructed compressed prompt."""

    path.write_text(
        prompt,
        encoding="utf-8"
    )


def main():

    print("Loading compressed information units...")

    units = load_compressed_units(INPUT_PATH)

    print(
        "Compressed information units:",
        len(units)
    )

    print("\nReconstructing compressed prompt...")

    compressed_prompt = reconstruct_prompt(units)

    save_compressed_prompt(
        compressed_prompt,
        OUTPUT_PATH
    )

    print("\n===================================")
    print("COMPRESSED PROMPT CREATED")
    print("===================================")

    print("\nCompressed Prompt:\n")

    print(compressed_prompt)

    print(
        "\nCompressed prompt saved to:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()