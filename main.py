import json
from pathlib import Path

from src.token_counter import count_tokens


PROMPT_PATH = Path("data/prompts/sample_prompt.txt")
OUTPUT_PATH = Path("data/sample_prompts.json")


def main():
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()

    token_count = count_tokens(prompt)

    result = {
        "id": "sample_001",
        "prompt": prompt,
        "token_count": token_count
    }

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Prompt loaded successfully.")
    print(f"Character count: {len(prompt)}")
    print(f"Token count: {token_count}")
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()