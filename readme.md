# Token-Efficient Prompt Compression for Large Language Models with Quality-Aware Evaluation

## Problem Statement

Large language model prompts can contain substantial contextual and instructional information, resulting in increased token usage, computational cost, and inference latency.

This project investigates a graph-based prompt compression framework that aims to reduce prompt token usage while preserving semantic information, task instructions, and downstream response quality.

## Objectives

- Represent prompt information using semantic graphs.
- Identify important nodes and relationships.
- Select a compressed semantic subgraph.
- Reconstruct a concise natural-language prompt.
- Compare the proposed method with baseline compression approaches.
- Evaluate token reduction and quality preservation.
- Analyze the compression-quality trade-off.

## Methodology

Original Prompt
→ Preprocessing
→ Semantic Graph Construction
→ Importance Scoring
→ Subgraph Selection
→ Prompt Reconstruction
→ LLM Inference
→ Quality Evaluation

## Project Structure

```text
src/        Source code
data/       Prompts and experimental results
notebooks/  Experiments and analysis
docs/       Research documentation and diagrams