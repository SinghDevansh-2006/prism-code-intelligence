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

    hf auth login

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
| NDCG@10 | 0.853100 |
| MRR@10 | 0.821803 |
| Recall@10 | 0.949535 |
| Recall@20 | 0.973440 |
| Recall@100 | 0.993625 |

These scores come from the repository's local evaluation pipeline over the AppsRetrieval test split and qrels.

No model weights, fusion weights, gate thresholds, or reranking parameters were tuned from these test results.

### Benchmark Verification

The values reported above are the canonical metrics recomputed directly from the exact ordered submission artifact.

Verify them with:

    python verify_submission.py

The verifier loads the official AppsRetrieval test qrels and independently computes NDCG@10, MRR@10, MRR@100, Recall@10, Recall@20, and Recall@100 from the submitted top-100 rankings.

Submission artifact SHA-256:

    519fe7b0ad8bb4ae80f1da5192ceea067bc1c68b74cad6328ee7ac8a2de644d4

Because the submission file contains an explicit ranking order, these artifact-derived metrics are the values used in the README and demo UI. Score-based evaluation can differ slightly when retrieval scores are tied because of tie-order conventions.

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
- Reranking is disabled by default; enabling the experimental reranker increases latency.
- The packaged runtime index contains 5,000 training documents. The reported test metrics use 3,765 queries against a separate 8,765-document benchmark corpus; they are not measurements of the packaged runtime index.
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

## CPU portability and packaging

The API and retriever defaults use CPU. Optional acceleration is explicit:
`PRISM_DEVICE=mps ./run_demo.sh` on supported Apple hardware, or
`PRISM_DEVICE=cuda ./run_demo.sh` with a compatible CUDA installation.
Unavailable requested accelerators fail with an actionable error. Reranking
remains disabled in the API unless `PRISM_ENABLE_RERANKER=1` is set.
`/health` reports both settings.

The launcher works from any directory on macOS/Linux and uses the repository
virtual environment. Set `PRISM_OPEN_BROWSER=0` on headless machines.
For Windows, activate the Python environment and run
`python -m uvicorn api:app --host 127.0.0.1 --port 8000` from this directory.

Docker (CPU): `docker build -t prism-ui .`, then
`docker run --rm -p 8000:8000 -e HF_TOKEN -v prism-hf:/root/.cache/huggingface prism-ui`.
Supply your own authorized Hugging Face token through the environment; tokens
and model caches are not included in the image. The first semantic search
downloads the gated models and requires network access and sufficient RAM.

## MTEB result export

Run `python export_mteb_results.py` to evaluate the unchanged frozen top-100
rankings using MTEB's retrieval metrics and serialize its `TaskResult.to_dict()`.
`submission/appsretrieval_results.json` is the MTEB result object; the original
`AppsRetrieval_inference_results.json` remains the raw ordered ranking artifact.
The accompanying provenance JSON records hashes, qrels revision, and method.
This is evaluation of saved rankings, not a new model inference run or an
end-to-end `mteb.evaluate` timing. Confidence/abstention metrics are omitted
because ordinal ranking scores cannot recover original model confidence.
MTEB rounds some metrics to five decimal places; the independently verified
six-decimal canonical metrics above remain unchanged.

## Repository contents and verification scope

`api.py`, `src/`, `web/`, `runtime_index/`, the launcher and dependency files
are the runnable application. `submission/` contains frozen rankings, the
MTEB result object and its provenance. `results/`, `models/`, `data/`, and
top-level benchmark, analysis, tuning and test scripts retain the experiment
evidence; they are optional for running the application and are deliberately
kept separate from the Docker image. Do not run every experiment as a setup step.

Experiment scripts now default to CPU and accept `PRISM_DEVICE=cpu`, `mps`,
or `cuda`. Run them from the repository root. Full embedding regeneration
can be slow and requires authorized model access, downloads and additional
cache storage. Existing frozen results are not regenerated by installation.

The GitHub Actions packaging check builds a fresh Linux CPU container and
tests health, bundled HTML, exact-usage and structural retrieval. It does not
validate gated semantic/evolution model downloads or reproduce benchmark
inference. Those still require the separate clean-machine verification.
Check the actual Actions outcome before claiming Docker build success.
