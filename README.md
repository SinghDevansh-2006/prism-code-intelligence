# PRISM Agentic Code Intelligence

Local agentic code retrieval built for the Samsung PRISM GenAI Hackathon 3.0 — Theme 1: Agentic Code Intelligence.

The system accepts natural-language questions about code and dynamically routes them through four complementary retrieval strategies:

- Semantic retrieval using EmbeddingGemma + Qwen3-Embedding
- Exact usage search using AST-derived symbol and import metadata
- Structural retrieval using scope-aware AST and call-order reasoning
- Evolutionary retrieval across multiple versions of a logical code artifact

The system focuses on retrieval: finding and ranking relevant code rather than generating replacement code.

---

## Quick Start

Python 3.11 is recommended.

Create the environment:

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

EmbeddingGemma is gated on Hugging Face. Make sure your account has access to:

    google/embeddinggemma-300m

Then authenticate locally:

    huggingface-cli login

Run the complete demo:

    ./run_demo.sh

Open:

    http://127.0.0.1:8000/

The launcher starts the FastAPI backend, pre-warms the semantic models, and opens the browser UI.

---

## Example Queries

Semantic:

    Find code that validates a tic-tac-toe board state

Exact usage:

    which files import math?

Structural:

    which functions call range before print?

Evolutionary:

    show binary search version history

---

## Architecture

    Natural-language query
            |
            v
    +----------------------+
    | Agentic Query Router |
    +----------------------+
       |       |       |       |
       |       |       |       +--> Evolutionary Retrieval
       |       |       +----------> Structural AST Search
       |       +------------------> Exact Usage Search
       +--------------------------> Semantic Retrieval
                                        |
                          +---------------------------+
                          | EmbeddingGemma            |
                          | Qwen3-Embedding-0.6B      |
                          +---------------------------+
                                        |
                              score normalization
                                        |
                                  65 / 35 fusion
                                        |
                                  ranked code

Semantic retrieval independently scores the corpus with both embedding models. Their score distributions are normalized per query and fused using the frozen weighting:

- 65% EmbeddingGemma
- 35% Qwen3-Embedding-0.6B

The engine also measures retrieval confidence. Confidence-gated cross-encoder reranking was experimentally evaluated, but dense fusion remains the screening retrieval configuration because reranking did not improve held-out AppsRetrieval NDCG@10.

---

## Retrieval Results

### Private Validation

The private validation split contains 1,000 queries sampled deterministically from the provided training data. Each validation query retrieves against all 5,000 training documents.

| System | NDCG@10 | MRR | Recall@10 |
| --- | ---: | ---: | ---: |
| Qwen3-Embedding-0.6B | 0.8215 | 0.7946 | 0.914 |
| EmbeddingGemma | 0.8716 | 0.8493 | 0.947 |
| Dual dense fusion | 0.9177 | 0.9004 | 0.973 |
| Fusion + selective reranker | 0.9204 | 0.9038 | 0.974 |

### AppsRetrieval Test Split

Frozen dense fusion evaluated over 3,765 AppsRetrieval test queries against all 8,765 corpus documents:

| Metric | Score |
| --- | ---: |
| NDCG@10 | 0.8535 |
| MRR@10 | 0.8223 |
| Recall@10 | 0.9495 |
| Recall@20 | 0.9734 |
| Recall@100 | 0.9936 |

These scores come from the repository's local evaluation pipeline over the AppsRetrieval test split and qrels.

No model weights, fusion weights, gate thresholds, or reranking parameters were tuned from these test results.

The ranked inference artifact is:

    submission/AppsRetrieval_inference_results.json

It contains all 3,765 test queries with the top 100 ranked corpus IDs for each query.

---

## Agentic Retrieval Routes

### 1. Semantic Retrieval

Used for descriptive intent such as:

    Find code that validates a tic-tac-toe board state

Execution flow:

    query
      -> EmbeddingGemma
      -> Qwen3-Embedding
      -> normalize score distributions
      -> 65/35 fusion
      -> confidence measurement
      -> ranked code snippets

### 2. Exact Usage Search

Used for explicit symbol, API, function, or import questions such as:

    which files import math?

The engine searches AST-derived metadata rather than depending on semantic similarity alone.

### 3. Structural Retrieval

Used for structural relationships such as:

    which functions call range before print?

The structural index stores scope-aware information including calls, imports, functions, classes, and line numbers so constraints can be verified structurally.

### 4. Evolutionary Retrieval

Used for version-history questions such as:

    show binary search version history

The version-aware index stores:

- logical code identity
- version
- embedding
- previous-version similarity
- change classification
- code

The included multi-version example is a controlled capability fixture rather than historical APPS version ground truth.

---

## API

Run manually:

    uvicorn api:app --host 127.0.0.1 --port 8000

Health endpoint:

    GET /health

Search endpoint:

    POST /search

Example request:

    curl -X POST http://127.0.0.1:8000/search \
      -H "Content-Type: application/json" \
      -d '{"query":"which functions call range before print?","top_k":5}'

The response includes:

- selected route
- routing reason
- router confidence
- execution strategy
- execution steps
- retrieval latency
- ranked code
- AST evidence when applicable
- version timeline when applicable

---

## Repository Structure

    api.py
    run_demo.sh
    requirements.txt
    README.md

    web/
        index.html

    src/
        agentic_engine.py
        query_router.py
        retriever.py
        structural_metadata.py
        structural_search.py
        versioned_retriever.py
        code_normalizer.py

    runtime_index/
        corpus.jsonl
        document_ids.json
        embeddinggemma_documents.npy
        qwen_documents.npy
        structural_index.jsonl
        versioned_semantic_test/

    results/

    submission/
        AppsRetrieval_inference_results.json

Additional benchmark, analysis, and test scripts are retained for reproducibility and ablation evidence.

---

## Key Design Decisions

### Complementary Dense Retrieval

EmbeddingGemma and Qwen make different retrieval errors. Score-level fusion improved private validation substantially over either embedding model independently.

### Routed AST Intelligence

Global AST-signature fusion was experimentally weaker, so AST evidence is activated when query intent requires structural or exact-usage reasoning.

### Evidence-Driven Feature Selection

BM25, global AST fusion, adaptive fusion, and selective reranking were all benchmarked rather than automatically added to the final stack.

Components that did not improve the target retrieval objective were not forced into the screening configuration.

### Local-First Execution

The prototype does not require a paid inference API. It was developed and tested locally on Apple Silicon.

### Incremental Version Retrieval

The evolutionary index is append-only. New versions can be embedded and added without rebuilding the entire version store.

---

## Reproducibility

Important experiment and evaluation scripts include:

    create_split.py
    benchmark_qwen.py
    benchmark_embeddinggemma.py
    benchmark_bm25.py
    benchmark_structural_signature.py
    fine_tune_dense_fusion.py
    derive_confidence_threshold.py
    evaluate_final_clean_pipeline.py
    prepare_official_embeddings.py
    evaluate_official_dense.py
    evaluate_official_final_pipeline.py

Large intermediate embedding caches are excluded from Git because they are reproducible and unnecessary for running the packaged demo.

---

## Demo Behavior

The FastAPI service lazy-loads retrieval engines.

A cold semantic request may take tens of seconds while pretrained models are loaded.

The provided run_demo.sh script pre-warms the semantic stack so judge-facing queries run against an already-loaded service.

Exact usage, structural, and evolutionary routes are lightweight after their indexes are available.

---

## Limitations

- Semantic model startup has a cold-load cost.
- Warm semantic latency depends on whether confidence-gated reranking is triggered.
- The packaged runtime corpus represents the hackathon retrieval corpus rather than a complete arbitrary-repository ingestion product.
- The included evolutionary demonstration uses a controlled multi-version fixture.
- Third-party pretrained models remain subject to their original licenses and access requirements.

---

## Model Dependencies

Primary pretrained model dependencies:

- google/embeddinggemma-300m
- Qwen/Qwen3-Embedding-0.6B
- mixedbread-ai/mxbai-rerank-xsmall-v1

Third-party model and dataset licenses and terms continue to apply.

---

## Submission Artifact

AppsRetrieval inference output:

    submission/AppsRetrieval_inference_results.json

Format:

    query_id -> ordered list of top-100 corpus document IDs

Queries:

    3,765

---

## Hackathon

Samsung PRISM GenAI Hackathon 3.0

Theme 1: Agentic Code Intelligence
