import json
from pathlib import Path


ORIGINAL_UNITS_PATH = Path(
    "data/preprocessed_prompts.json"
)

COMPRESSED_UNITS_PATH = Path(
    "data/compressed_information_units.json"
)


def load_units(path):
    """Load information units from JSON."""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def calculate_instruction_fidelity(
    original_units,
    compressed_units
):
    """
    Calculate the proportion of critical information
    units retained after compression.

    Critical units include:
    - role
    - instruction
    - constraint
    """

    critical_types = {
        "role",
        "instruction",
        "constraint"
    }

    original_critical_units = [
        unit
        for unit in original_units
        if unit.get(
            "text_type",
            ""
        ).lower() in critical_types
    ]

    compressed_ids = {
        unit["original_id"]
        for unit in compressed_units
    }

    retained_critical_units = [
        unit
        for index, unit in enumerate(
            original_units
        )
        if (
            index in compressed_ids
            and unit.get(
                "text_type",
                ""
            ).lower() in critical_types
        )
    ]

    total_critical = len(
        original_critical_units
    )

    retained_critical = len(
        retained_critical_units
    )

    if total_critical == 0:
        return (
            1.0,
            total_critical,
            retained_critical
        )

    fidelity = (
        retained_critical /
        total_critical
    )

    return (
        fidelity,
        total_critical,
        retained_critical
    )


def main():

    print(
        "Loading original information units..."
    )

    original_units = load_units(
        ORIGINAL_UNITS_PATH
    )

    print(
        "Loading compressed information units..."
    )

    compressed_units = load_units(
        COMPRESSED_UNITS_PATH
    )

    print(
        "\nCalculating instruction fidelity..."
    )

    (
        fidelity,
        total_critical,
        retained_critical
    ) = calculate_instruction_fidelity(
        original_units,
        compressed_units
    )

    print(
        "\n==================================="
    )

    print(
        "INSTRUCTION FIDELITY EVALUATION"
    )

    print(
        "==================================="
    )

    print(
        f"Original critical units: "
        f"{total_critical}"
    )

    print(
        f"Retained critical units: "
        f"{retained_critical}"
    )

    print(
        f"Instruction fidelity: "
        f"{fidelity:.4f}"
    )

    print(
        f"Instruction fidelity (%): "
        f"{fidelity * 100:.2f}%"
    )


if __name__ == "__main__":
    main()