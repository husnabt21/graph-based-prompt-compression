import json
from pathlib import Path

import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_PATH = Path("data/semantic_graph.json")
OUTPUT_PATH = Path("data/compressed_information_units.json")

MODEL_NAME = "all-MiniLM-L6-v2"

# Target proportion of information units to retain
RETENTION_RATIO = 0.80

# Similarity above this value is treated as redundancy
REDUNDANCY_THRESHOLD = 0.85


def load_graph(path):
    """Load semantic graph from JSON."""

    with open(path, "r", encoding="utf-8") as file:
        graph_data = json.load(file)

    return nx.node_link_graph(
        graph_data,
        edges="links"
    )


def generate_embeddings(graph):
    """
    Generate sentence embeddings for all information units.
    """

    texts = [
        graph.nodes[node]["text"]
        for node in graph.nodes()
    ]

    model = SentenceTransformer(
        MODEL_NAME
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


def calculate_redundancy_scores(embeddings):
    """
    Calculate semantic redundancy for every information unit.

    A unit is considered redundant when it has a highly
    similar counterpart in the prompt.
    """

    similarity_matrix = cosine_similarity(
        embeddings
    )

    redundancy_scores = {}

    number_of_units = len(
        embeddings
    )

    for i in range(number_of_units):

        similarities = []

        for j in range(number_of_units):

            if i == j:
                continue

            similarities.append(
                similarity_matrix[i][j]
            )

        if similarities:

            max_similarity = max(
                similarities
            )

        else:

            max_similarity = 0.0

        if max_similarity >= REDUNDANCY_THRESHOLD:

            redundancy_scores[i] = (
                max_similarity
            )

        else:

            redundancy_scores[i] = 0.0

    return redundancy_scores


def calculate_node_importance(
    graph,
    redundancy_scores
):
    """
    Calculate quality-aware graph importance.

    Components:

    1. Graph centrality
    2. Semantic connectivity
    3. Information type
    4. Role/constraint protection
    5. Redundancy penalty
    """

    degree_centrality = (
        nx.degree_centrality(graph)
    )

    weighted_degrees = {}

    for node in graph.nodes():

        weighted_degree = sum(
            data.get("weight", 0.0)
            for _, _, data in graph.edges(
                node,
                data=True
            )
        )

        weighted_degrees[node] = (
            weighted_degree
        )

    max_weighted_degree = max(
        weighted_degrees.values(),
        default=1.0
    )

    if max_weighted_degree == 0:

        max_weighted_degree = 1.0

    importance_scores = {}

    for node in graph.nodes():

        text_type = graph.nodes[node].get(
            "text_type",
            "unknown"
        ).lower()

        # -----------------------------------------
        # Graph centrality
        # -----------------------------------------

        centrality_score = (
            degree_centrality[node]
        )

        # -----------------------------------------
        # Weighted semantic connectivity
        # -----------------------------------------

        weighted_degree = (
            weighted_degrees[node]
        )

        normalized_connectivity = (
            weighted_degree /
            max_weighted_degree
        )

        # -----------------------------------------
        # Information type
        # -----------------------------------------

        if text_type == "role":

            type_score = 1.00

        elif text_type == "constraint":

            type_score = 1.00

        elif text_type == "instruction":

            type_score = 0.85

        elif text_type == "context":

            type_score = 0.60

        else:

            type_score = 0.50

        # -----------------------------------------
        # Protection score
        # -----------------------------------------

        if text_type in (
            "role",
            "constraint"
        ):

            protection_score = 1.00

        else:

            protection_score = 0.00

        # -----------------------------------------
        # Redundancy
        # -----------------------------------------

        redundancy = (
            redundancy_scores.get(
                node,
                0.0
            )
        )

        # -----------------------------------------
        # Final importance
        # -----------------------------------------

        final_score = (
            0.25 * centrality_score
            + 0.30 * normalized_connectivity
            + 0.25 * type_score
            + 0.20 * protection_score
            - 0.05 * redundancy
        )

        importance_scores[node] = {

            "degree_centrality":
                centrality_score,

            "weighted_degree":
                weighted_degree,

            "normalized_connectivity":
                normalized_connectivity,

            "type_score":
                type_score,

            "protection_score":
                protection_score,

            "redundancy_score":
                redundancy,

            "final_score":
                final_score
        }

    return importance_scores


def select_nodes(
    graph,
    importance_scores,
    retention_ratio=RETENTION_RATIO
):
    """
    Select information units using a quality-aware strategy.

    Role and constraint units are mandatory.

    Remaining capacity is filled using the highest
    quality graph-based importance scores.
    """

    total_nodes = graph.number_of_nodes()

    target_nodes = max(
        1,
        round(
            total_nodes *
            retention_ratio
        )
    )

    # -----------------------------------------
    # Identify mandatory units
    # -----------------------------------------

    mandatory_nodes = []

    candidate_nodes = []

    for node in graph.nodes():

        text_type = graph.nodes[node].get(
            "text_type",
            "unknown"
        ).lower()

        if text_type in (
            "role",
            "constraint"
        ):

            mandatory_nodes.append(node)

        else:

            candidate_nodes.append(node)

    # -----------------------------------------
    # Rank candidate units
    # -----------------------------------------

    candidate_nodes.sort(
        key=lambda node:
        importance_scores[node][
            "final_score"
        ],
        reverse=True
    )

    # -----------------------------------------
    # Start with mandatory information
    # -----------------------------------------

    selected_nodes = list(
        mandatory_nodes
    )

    # -----------------------------------------
    # Fill remaining capacity
    # -----------------------------------------

    remaining_capacity = (
        target_nodes -
        len(selected_nodes)
    )

    if remaining_capacity > 0:

        selected_nodes.extend(
            candidate_nodes[
                :remaining_capacity
            ]
        )

    # -----------------------------------------
    # If mandatory units exceed target,
    # preserve them anyway.
    # -----------------------------------------

    selected_nodes = sorted(
        set(selected_nodes)
    )

    return selected_nodes


def save_compressed_units(
    graph,
    selected_nodes,
    path
):
    """Save selected information units."""

    compressed_units = []

    for node in selected_nodes:

        data = graph.nodes[node]

        compressed_units.append(
            {
                "original_id": node,
                "text": data["text"],
                "text_type": data.get(
                    "text_type",
                    "unknown"
                )
            }
        )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            compressed_units,
            file,
            indent=2,
            ensure_ascii=False
        )


def print_summary(
    graph,
    selected_nodes,
    importance_scores
):
    """Print compression results."""

    total_nodes = (
        graph.number_of_nodes()
    )

    retained_nodes = len(
        selected_nodes
    )

    retention = (
        retained_nodes /
        total_nodes
    )

    reduction = (
        1 -
        retention
    )

    print(
        "\n==================================="
    )

    print(
        "QUALITY-AWARE GRAPH COMPRESSION"
    )

    print(
        "==================================="
    )

    print(
        f"Original information units: "
        f"{total_nodes}"
    )

    print(
        f"Retained information units: "
        f"{retained_nodes}"
    )

    print(
        f"Retention ratio: "
        f"{retention:.4f}"
    )

    print(
        f"Information-unit reduction: "
        f"{reduction * 100:.2f}%"
    )

    print("\nNode importance:")

    ranked_nodes = sorted(
        importance_scores.items(),
        key=lambda item:
        item[1]["final_score"],
        reverse=True
    )

    for node, scores in ranked_nodes:

        text_type = graph.nodes[node].get(
            "text_type",
            "unknown"
        )

        print(
            f"Node {node}: "
            f"final={scores['final_score']:.4f}, "
            f"connectivity="
            f"{scores['normalized_connectivity']:.4f}, "
            f"protection="
            f"{scores['protection_score']:.1f}, "
            f"redundancy="
            f"{scores['redundancy_score']:.4f}, "
            f"type={text_type}"
        )

    print("\nSelected nodes:")

    for node in selected_nodes:

        print(
            f"Node {node}: "
            f"{graph.nodes[node]['text']}"
        )


def main():

    print(
        "Loading semantic graph..."
    )

    graph = load_graph(
        INPUT_PATH
    )

    print(
        f"Graph loaded: "
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    print(
        "\nGenerating embeddings for "
        "redundancy analysis..."
    )

    embeddings = generate_embeddings(
        graph
    )

    print(
        f"Embeddings generated: "
        f"{embeddings.shape}"
    )

    print(
        "\nCalculating semantic redundancy..."
    )

    redundancy_scores = (
        calculate_redundancy_scores(
            embeddings
        )
    )

    print(
        "\nCalculating quality-aware "
        "node importance..."
    )

    importance_scores = (
        calculate_node_importance(
            graph,
            redundancy_scores
        )
    )

    print(
        "\nSelecting important "
        "information units..."
    )

    selected_nodes = select_nodes(
        graph,
        importance_scores,
        RETENTION_RATIO
    )

    save_compressed_units(
        graph,
        selected_nodes,
        OUTPUT_PATH
    )

    print_summary(
        graph,
        selected_nodes,
        importance_scores
    )

    print(
        "\nCompressed units saved to:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()