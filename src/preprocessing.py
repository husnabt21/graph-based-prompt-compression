import re
import json
from pathlib import Path

def clean_text(text):
    """
    Basic text cleaning.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text

def segment_sentences(text):
    """
    Split text into individual sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def classify_unit(text):
    """
    Basic rule-based classification of an information unit.
    """

    text_lower = text.lower()

    instruction_words = [
        "analyze",
        "explain",
        "compare",
        "describe",
        "identify",
        "evaluate",
        "summarize",
        "discuss",
        "provide",
        "present"
    ]

    constraint_words = [
        "must",
        "should",
        "required",
        "only",
        "at least",
        "maximum",
        "minimum",
        "do not",
        "using"
    ]

    if any(word in text_lower for word in instruction_words):
        return "instruction"

    if any(word in text_lower for word in constraint_words):
        return "constraint"

    return "context"

def create_information_units(text):
    """
    Convert a prompt into structured information units.
    """

    cleaned_text = clean_text(text)
    sentences = segment_sentences(cleaned_text)

    units = []

    for index, sentence in enumerate(sentences, start=1):

        unit = {
            "id": f"unit_{index:03d}",
            "text": sentence,
            "type": classify_unit(sentence)
        }

        units.append(unit)

    return units

def save_information_units(units, output_path):
    """
    Save information units to a JSON file.
    """

    Path(output_path).write_text(
        json.dumps(units, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

if __name__ == "__main__":

    input_path = Path("data/prompts/sample_prompt.txt")
    output_path = Path("data/preprocessed_prompts.json")

    # Load the original prompt
    prompt = input_path.read_text(encoding="utf-8")

    # Preprocess the prompt and create information units
    units = create_information_units(prompt)

    # Save the processed information units
    save_information_units(units, output_path)

    # Display results
    print("Prompt preprocessing completed successfully.")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Extracted information units: {len(units)}")

    print("\nInformation Units:")

    for unit in units:
        print(
            f"{unit['id']} | "
            f"{unit['type']} | "
            f"{unit['text']}"
        )