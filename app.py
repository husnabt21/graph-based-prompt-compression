import sys
from pathlib import Path

import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

sys.path.append(str(SRC_DIR))

from preprocessing import create_information_units
from semantic_graph import build_semantic_graph
from graph_compressor import (
    calculate_redundancy_scores,
    calculate_node_importance,
    select_nodes
)


MODEL_NAME = "all-MiniLM-L6-v2"
RETENTION_RATIO = 0.80


st.set_page_config(
    page_title="Token-Efficient Prompt Compression",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def load_embedding_model():
    """
    Load the sentence-transformer model once
    and reuse it across Streamlit runs.
    """

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME
    )


st.title(
    "Token-Efficient Prompt Compression"
)

st.subheader(
    "Graph-Based Prompt Compression with "
    "Quality-Aware Evaluation"
)

st.write(
    "Enter a prompt below to analyze its semantic "
    "structure and generate a compressed version "
    "using graph-based information selection."
)


prompt = st.text_area(
    "Enter your prompt:",
    height=250,
    placeholder="Paste your prompt here..."
)


if st.button("Compress Prompt"):

    if not prompt.strip():

        st.warning(
            "Please enter a prompt first."
        )

    else:

        with st.spinner(
            "Analyzing and compressing prompt..."
        ):

            # Step 1: Preprocess prompt
            information_units = (
                create_information_units(
                    prompt
                )
            )

            # Step 2: Load cached embedding model
            model = load_embedding_model()

            texts = [
                unit["text"]
                for unit in information_units
            ]

            # Step 3: Generate embeddings
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # Step 4: Build semantic graph
            graph = build_semantic_graph(
                information_units,
                embeddings
            )

            # Step 5: Calculate redundancy
            redundancy_scores = (
                calculate_redundancy_scores(
                    embeddings
                )
            )

            # Step 6: Calculate node importance
            importance_scores = (
                calculate_node_importance(
                    graph,
                    redundancy_scores
                )
            )

            # Step 7: Select important nodes
            selected_nodes = select_nodes(
                graph,
                importance_scores,
                RETENTION_RATIO
            )

            # Step 8: Create compressed prompt
            compressed_prompt = " ".join(
                graph.nodes[node]["text"]
                for node in selected_nodes
            )

            # ==========================================
            # EVALUATION METRICS
            # ==========================================

            tokenizer = tiktoken.get_encoding(
                "cl100k_base"
            )

            original_tokens = len(
                 tokenizer.encode(prompt)
            )

            compressed_tokens = len(
               tokenizer.encode(compressed_prompt)
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
                original_tokens / compressed_tokens if compressed_tokens > 0 else 0.0
            )

            compressed_embedding = model.encode(
                [compressed_prompt],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            original_embedding = model.encode(
                [prompt],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            semantic_similarity = float(
                cosine_similarity(
                    original_embedding,
                    compressed_embedding
                )[0][0]
            )


        st.success(
            "Prompt compression completed successfully."
        )


        # ==========================================
        # COMPRESSION SUMMARY
        # ==========================================

        st.subheader(
            "Compression Summary"
        )

        col1, col2, col3, col4 = st.columns(4)

        total_units = len(
            information_units
        )

        retained_units = len(
            selected_nodes
        )

        reduction = (
            1 -
            (
                retained_units /
                total_units
            )
        ) * 100

        col1.metric(
            "Original Units",
            total_units
        )

        col2.metric(
            "Retained Units",
            retained_units
        )

        col3.metric(
            "Unit Reduction",
            f"{reduction:.2f}%"
        )

        col4.metric(
            "Graph Edges",
            graph.number_of_edges()
        )


        # ==========================================
        # ORIGINAL PROMPT
        # ==========================================

        st.subheader(
            "Original Prompt"
        )

        st.text_area(
            "Original",
            prompt,
            height=220,
            disabled=True
        )


        # ==========================================
        # COMPRESSED PROMPT
        # ==========================================

        st.subheader(
            "Compressed Prompt"
        )

        st.text_area(
            "Compressed",
            compressed_prompt,
            height=220,
            disabled=True
        )


        # ==========================================
        # SELECTED INFORMATION UNITS
        # ==========================================

        st.subheader(
            "Selected Information Units"
        )

        for node in selected_nodes:

            node_data = graph.nodes[node]

            st.write(
                f"**Node {node} — "
                f"{node_data.get('text_type', 'unknown')}**"
            )

            st.write(
                node_data["text"]
            )

            st.divider()


        # ==========================================
        # SEMANTIC GRAPH VISUALIZATION
        # ==========================================

        st.subheader(
            "Semantic Graph"
        )

        st.write(
            "Each node represents an information unit "
            "from the original prompt. Edges represent "
            "semantic similarity between information units."
        )

        st.write(
            f"Nodes: **{graph.number_of_nodes()}**"
        )

        st.write(
            f"Edges: **{graph.number_of_edges()}**"
        )


        # Create graph layout
        positions = nx.spring_layout(
            graph,
            seed=42,
            k=1.5,
            scale=0.8
        )


        # Create figure
        fig, ax = plt.subplots(
            figsize=(8,3)
        )

        

        ax.set_xmargin(0.15)
        ax.set_ymargin(0.15)

        # Separate retained and removed nodes
        retained_nodes = set(
            selected_nodes
        )

        removed_nodes = [
            node
            for node in graph.nodes()
            if node not in retained_nodes
        ]


        # Draw all graph edges
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            width=1.5,
            alpha=0.6
        )


        # Draw retained nodes
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=selected_nodes,
            ax=ax,
            node_size=900,
            alpha=0.9
        )


        # Draw removed nodes
        if removed_nodes:

            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=removed_nodes,
                ax=ax,
                node_size=900,
                alpha=0.4
            )


        # Node labels
        labels = {
            node: f"Node {node}"
            for node in graph.nodes()
        }

        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            ax=ax,
            font_size=10
        )


        ax.set_title(
            "Semantic Graph of Prompt Information Units"
        )

        ax.axis("off")

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )


        # ==========================================
        # GRAPH NODE DETAILS
        # ==========================================

        st.subheader(
            "Graph Node Details"
        )

        for node in graph.nodes():

            node_data = graph.nodes[node]

            status = (
                "Retained"
                if node in retained_nodes
                else "Removed"
            )

            st.write(
                f"**Node {node}** | "
                f"Type: {node_data.get('text_type', 'unknown')} | "
                f"Status: {status}"
            )

            st.caption(
                node_data["text"]
            )



        st.subheader(
            "Evaluation Metrics"
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Original Tokens",
            original_tokens
        )

        metric2.metric(
            "Compressed Tokens",
            compressed_tokens
        )

        metric3.metric(
            "Tokens Reduced",
            tokens_reduced
        )


        metric4, metric5, metric6 = st.columns(3)

        metric4.metric(
            "Token Reduction",
            f"{token_reduction:.2f}%"
        )

        metric5.metric(
            "Compression Ratio",
            f"{compression_ratio:.2f}x"
        )

        metric6.metric(
            "Semantic Similarity",
            f"{semantic_similarity:.4f}"
        )

        # ==========================================
        # INSTRUCTION FIDELITY
        # ==========================================

        critical_units = [
            unit
            for unit in information_units
            if unit.get("text_type", "").lower()
            in ("role", "constraint", "instruction")
        ]

        retained_unit_ids = set(
            selected_nodes
        )

        retained_critical_units = [
            unit
            for index, unit in enumerate(information_units)
            if index in retained_unit_ids
            and unit.get("text_type", "").lower()
            in ("role", "constraint", "instruction")
        ]

        if critical_units:

            instruction_fidelity = (
                len(retained_critical_units)
                /
                len(critical_units)
            )

        else:

            instruction_fidelity = 1.0


        st.metric(
            "Instruction Fidelity",
            f"{instruction_fidelity * 100:.2f}%"
        )