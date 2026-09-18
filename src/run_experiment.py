import json
from pathlib import Path

import pandas as pd
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import create_information_units
from semantic_graph import build_semantic_graph
from graph_compressor import (
    calculate_redundancy_scores,
    calculate_node_importance,
    select_nodes
)


MODEL_NAME = "all-MiniLM-L6-v2"
RETENTION_RATIO = 0.80

INPUT_PATH = Path(
    "data/experiment/evaluation_prompts.json"
)

OUTPUT_PATH = Path(
    "data/final_results.csv"
)


def main():

    print("Loading experiment prompts...")

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        prompts = json.load(file)


    print(
        f"Prompts loaded: {len(prompts)}"
    )


    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    tokenizer = tiktoken.get_encoding(
        "cl100k_base"
    )

    results = []


    for item in prompts:

        prompt_id = item["id"]
        category = item["category"]
        prompt = item["prompt"]


        print(
            f"\nProcessing {prompt_id} "
            f"({category})..."
        )


        # ======================================
        # PREPROCESSING
        # ======================================

        information_units = (
            create_information_units(
                prompt
            )
        )


        # ======================================
        # EMBEDDINGS
        # ======================================

        texts = [
            unit["text"]
            for unit in information_units
        ]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )


        # ======================================
        # SEMANTIC GRAPH
        # ======================================

        graph = build_semantic_graph(
            information_units,
            embeddings
        )


        # ======================================
        # QUALITY-AWARE COMPRESSION
        # ======================================

        redundancy_scores = (
            calculate_redundancy_scores(
                embeddings
            )
        )

        importance_scores = (
            calculate_node_importance(
                graph,
                redundancy_scores
            )
        )

        selected_nodes = select_nodes(
            graph,
            importance_scores,
            RETENTION_RATIO
        )


        compressed_prompt = " ".join(
            graph.nodes[node]["text"]
            for node in selected_nodes
        )


        # ======================================
        # TOKEN METRICS
        # ======================================

        original_tokens = len(
            tokenizer.encode(prompt)
        )

        compressed_tokens = len(
            tokenizer.encode(
                compressed_prompt
            )
        )

        tokens_reduced = (
            original_tokens -
            compressed_tokens
        )

        token_reduction = (
            tokens_reduced /
            original_tokens
        ) * 100

        compression_ratio = (
            original_tokens /
            compressed_tokens
        )


        # ======================================
        # SEMANTIC SIMILARITY
        # ======================================

        original_embedding = model.encode(
            [prompt],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        compressed_embedding = model.encode(
            [compressed_prompt],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        semantic_similarity = float(
            cosine_similarity(
                original_embedding,
                compressed_embedding
            )[0][0]
        )


        # ======================================
        # INSTRUCTION FIDELITY
        # ======================================

        critical_units = [
            unit
            for unit in information_units
            if unit.get("text_type", "").lower()
            in (
                "role",
                "constraint",
                "instruction"
            )
        ]

        retained_critical_units = [
            node
            for node in selected_nodes
            if graph.nodes[node].get(
                "text_type",
                ""
            ).lower()
            in (
                "role",
                "constraint",
                "instruction"
            )
        ]

        if critical_units:

            instruction_fidelity = (
                len(retained_critical_units)
                /
                len(critical_units)
            ) * 100

        else:

            instruction_fidelity = 100.0


        # ======================================
        # SAVE RESULT
        # ======================================

        results.append(
            {
                "prompt_id": prompt_id,
                "category": category,
                "original_units": len(
                    information_units
                ),
                "retained_units": len(
                    selected_nodes
                ),
                "graph_edges": graph.number_of_edges(),
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "tokens_reduced": tokens_reduced,
                "token_reduction_percent":
                    round(
                        token_reduction,
                        2
                    ),
                "compression_ratio":
                    round(
                        compression_ratio,
                        4
                    ),
                "semantic_similarity":
                    round(
                        semantic_similarity,
                        4
                    ),
                "instruction_fidelity_percent":
                    round(
                        instruction_fidelity,
                        2
                    )
            }
        )


        print(
            f"Original tokens: "
            f"{original_tokens}"
        )

        print(
            f"Compressed tokens: "
            f"{compressed_tokens}"
        )

        print(
            f"Token reduction: "
            f"{token_reduction:.2f}%"
        )

        print(
            f"Semantic similarity: "
            f"{semantic_similarity:.4f}"
        )

        print(
            f"Instruction fidelity: "
            f"{instruction_fidelity:.2f}%"
        )


    # ==========================================
    # CREATE DATAFRAME
    # ==========================================

    dataframe = pd.DataFrame(
        results
    )


    # ==========================================
    # SAVE CSV
    # ==========================================

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        "\n===================================="
    )

    print(
        "EXPERIMENT COMPLETED"
    )

    print(
        "===================================="
    )

    print(
        f"Results saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Total prompts: "
        f"{len(dataframe)}"
    )


if __name__ == "__main__":

    main()