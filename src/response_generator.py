from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "google/flan-t5-small"

BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_PROMPT_PATH = (
    BASE_DIR / "data" / "prompts" / "sample_prompt.txt"
)

COMPRESSED_PROMPT_PATH = (
    BASE_DIR / "data" / "compressed_prompt.txt"
)

RESULTS_DIR = BASE_DIR / "data" / "results"

ORIGINAL_RESPONSE_PATH = (
    RESULTS_DIR / "original_response.txt"
)

COMPRESSED_RESPONSE_PATH = (
    RESULTS_DIR / "compressed_response.txt"
)


# =========================================================
# Load model
# =========================================================

def load_model():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading language model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    # Use GPU if available, otherwise CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"Model loaded successfully.")
    print(f"Device: {device}")

    return tokenizer, model, device


# =========================================================
# Load prompt
# =========================================================

def load_prompt(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read().strip()


# =========================================================
# Generate response
# =========================================================

def generate_response(
    tokenizer,
    model,
    device,
    prompt
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            min_new_tokens=60,
            do_sample=False,
            num_beams=5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response.strip()


# =========================================================
# Save response
# =========================================================

def save_response(path, response):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(response)


# =========================================================
# Main pipeline
# =========================================================

def main():

    print("=" * 60)
    print("RESPONSE GENERATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Load prompts
    # -----------------------------------------------------

    print("\nLoading prompts...")

    original_prompt = load_prompt(
        ORIGINAL_PROMPT_PATH
    )

    compressed_prompt = load_prompt(
        COMPRESSED_PROMPT_PATH
    )

    print("Original prompt loaded.")
    print("Compressed prompt loaded.")

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    tokenizer, model, device = load_model()

    # -----------------------------------------------------
    # Generate original response
    # -----------------------------------------------------

    print(
        "\nGenerating response for original prompt..."
    )

    original_response = generate_response(
        tokenizer,
        model,
        device,
        original_prompt
    )

    print("Original response generated.")

    # -----------------------------------------------------
    # Generate compressed response
    # -----------------------------------------------------

    print(
        "\nGenerating response for compressed prompt..."
    )

    compressed_response = generate_response(
        tokenizer,
        model,
        device,
        compressed_prompt
    )

    print("Compressed response generated.")

    # -----------------------------------------------------
    # Save responses
    # -----------------------------------------------------

    save_response(
        ORIGINAL_RESPONSE_PATH,
        original_response
    )

    save_response(
        COMPRESSED_RESPONSE_PATH,
        compressed_response
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nResponses saved successfully.")

    print(
        f"Original response: "
        f"{ORIGINAL_RESPONSE_PATH}"
    )

    print(
        f"Compressed response: "
        f"{COMPRESSED_RESPONSE_PATH}"
    )

    print("\n" + "=" * 60)
    print("RESPONSE GENERATION COMPLETED")
    print("=" * 60)


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()
