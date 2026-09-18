from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_PROMPT_PATH = (
    BASE_DIR / "data" / "prompts" / "sample_prompt.txt"
)

OUTPUT_PATH = (
    BASE_DIR / "data" / "baseline_compressed_prompt.txt"
)


def main():

    print("=" * 60)
    print("BASELINE PROMPT COMPRESSION")
    print("=" * 60)

    with open(
        ORIGINAL_PROMPT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        prompt = file.read().strip()

    sentences = [
        sentence.strip()
        for sentence in prompt.split(".")
        if sentence.strip()
    ]

    # Simple baseline:
    # retain approximately 80% of sentences
    keep_count = max(
        1,
        round(len(sentences) * 0.80)
    )

    selected = sentences[:keep_count]

    compressed = ". ".join(selected) + "."

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(compressed)

    print(f"\nOriginal sentences: {len(sentences)}")
    print(f"Baseline sentences: {keep_count}")

    print("\nBaseline compressed prompt:")
    print(compressed)

    print(f"\nSaved to: {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("BASELINE COMPRESSION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()