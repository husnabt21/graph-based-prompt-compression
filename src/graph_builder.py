from pathlib import Path
import json

import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.50


def load_information_units(input_path: Path) -> list[dict]:
    """
    Load preprocessed information units from a JSON file.
    """

    with open(input_path, "r", encoding="utf-8") as file:
        units = json.load(file)

    if not isinstance(units, list):
        raise ValueError("Expected JSON file to contain a list of information units.")

    return units


def validate_information_units(units: list[dict]) -> None:
    """
    Validate that every information unit contains
    the fields required for graph construction.
    """

    required_fields = {"id", "text", "type"}

    for unit in units:
        missing_fields = required_fields - unit.keys()

        if missing_fields:
            raise ValueError(
                f"Information unit {unit.get('id', 'unknown')} "
                f"is missing fields: {missing_fields}"
            )


def create_embeddings(
    units: list[dict],
    model_name: str = DEFAULT_MODEL
):
    """
    Generate semantic embeddings for each information unit.
    """

    model = SentenceTransformer(model_name)

    texts = [unit["text"] for unit in units]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


def calculate_similarity_matrix(embeddings):
    """
    Calculate cosine similarity between every pair of information units.
    """

    similarity_matrix = cosine_similarity(embeddings)

    return similarity_matrix


def build_semantic_graph(
    units: list[dict],
    similarity_matrix,
    threshold: float = DEFAULT_THRESHOLD
) -> nx.Graph:
    """
    Construct an undirected weighted semantic graph.

    Nodes represent information units.
    Edges represent semantic relationships.
    Edge weights represent cosine similarity.
    """

    graph = nx.Graph()

    # Add information units as graph nodes
    for unit in units:
        graph.add_node(
            unit["id"],
            text=unit["text"],
            type=unit["type"]
        )

    # Add edges based on semantic similarity
    for i in range(len(units)):
        for j in range(i + 1, len(units)):

            similarity = float(similarity_matrix[i][j])

            if similarity >= threshold:
                graph.add_edge(
                    units[i]["id"],
                    units[j]["id"],
                    weight=similarity
                )

    return graph


def save_graph(graph: nx.Graph, output_path: Path) -> None:
    """
    Save the graph structure as JSON.
    """

    graph_data = nx.node_link_data(graph)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            graph_data,
            file,
            indent=4,
            ensure_ascii=False
        )


def build_graph_from_file(
    input_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD
) -> nx.Graph:
   
    units = load_information_units(input_path)

    validate_information_units(units)

    embeddings = create_embeddings(
        units,
        model_name=model_name
    )

    similarity_matrix = calculate_similarity_matrix(
        embeddings
    )

    graph = build_semantic_graph(
        units,
        similarity_matrix,
        threshold=threshold
    )

    save_graph(
        graph,
        output_path
    )

    return graph

if __name__ == "__main__":

    input_path = Path(
        "data/preprocessed_prompts.json"
    )

    output_path = Path(
        "data/semantic_graph.json"
    )

    graph = build_graph_from_file(
        input_path=input_path,
        output_path=output_path,
        threshold=0.50
    )

    print("\nSemantic Graph Construction")
    print("-" * 40)

    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")

    print("\nNodes:")

    for node, attributes in graph.nodes(data=True):
        print(
            f"{node} | "
            f"type={attributes['type']} | "
            f"text={attributes['text']}"
        )

    print("\nEdges:")

    for source, target, attributes in graph.edges(data=True):
        print(
            f"{source} <--> {target} | "
            f"similarity={attributes['weight']:.4f}"
        )