import json
from pathlib import Path

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.50

INPUT_PATH = Path("data/preprocessed_prompts.json")
OUTPUT_PATH = Path("data/semantic_graph.json")


def load_information_units(path):
    """Load information units from JSON."""

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Handle either a direct list or a dictionary containing units
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "information_units" in data:
            return data["information_units"]

        if "units" in data:
            return data["units"]

    raise ValueError("Could not find information units in JSON file.")


def generate_embeddings(information_units):
    """Generate semantic embeddings for information units."""

    texts = [unit["text"] for unit in information_units]

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


def build_semantic_graph(information_units, embeddings):
    """Build graph using cosine semantic similarity."""

    graph = nx.Graph()

    # Add nodes
    for i, unit in enumerate(information_units):

        graph.add_node(
            i,
            text=unit["text"],
            text_type=unit.get("text_type", "unknown")
        )

    # Calculate similarities
    similarity_matrix = cosine_similarity(embeddings)

    # Add edges
    for i in range(len(information_units)):

        for j in range(i + 1, len(information_units)):

            similarity = float(similarity_matrix[i][j])

            if similarity >= SIMILARITY_THRESHOLD:

                graph.add_edge(
                    i,
                    j,
                    weight=similarity
                )

    return graph


def save_graph(graph, path):
    """Save graph to JSON."""

    graph_data = nx.node_link_data(graph, edges="links")

    with open(path, "w", encoding="utf-8") as file:
        json.dump(graph_data, file, indent=2)


def print_graph_summary(graph, embeddings):

    print("\n===================================")
    print("SEMANTIC GRAPH CREATED SUCCESSFULLY")
    print("===================================")

    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    print("Embedding dimensions:", embeddings.shape)

    print("Similarity threshold:", SIMILARITY_THRESHOLD)

    print("\nNodes:")

    for node, data in graph.nodes(data=True):

        print(
            f"Node {node}: "
            f"{data['text']}"
        )

    print("\nEdges:")

    for source, target, data in graph.edges(data=True):

        print(
            f"{source} <--> {target} "
            f"(similarity={data['weight']:.4f})"
        )


def main():

    print("Loading information units...")

    information_units = load_information_units(INPUT_PATH)

    print(
        "Information units loaded:",
        len(information_units)
    )

    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(information_units)

    print(
        "Embeddings generated:",
        embeddings.shape
    )

    print("\nBuilding semantic graph...")

    graph = build_semantic_graph(
        information_units,
        embeddings
    )

    save_graph(graph, OUTPUT_PATH)

    print_graph_summary(
        graph,
        embeddings
    )

    print(
        "\nGraph saved to:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()