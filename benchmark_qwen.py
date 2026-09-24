import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_loader import load_dev_benchmark
from src.code_normalizer import compact_large_literals
from src.metrics import evaluate_rankings


MODEL = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 4096

CACHE_DIR = Path("cache/qwen06b")
RESULTS_DIR = Path("results")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("QWEN3-EMBEDDING-0.6B DEVELOPMENT BENCHMARK")
print("=" * 70)

print("\nLoading benchmark...")
queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]

query_texts = [row["text"] for row in queries]

print("Normalizing code...")
doc_texts = [
    compact_large_literals(row["text"])
    for row in corpus
]

print("\nLoading model...")
model = SentenceTransformer(
    MODEL,
    device="mps"
)
model.max_seq_length = MAX_SEQ_LENGTH


# --------------------------------------------------
# DOCUMENT EMBEDDINGS
# --------------------------------------------------

doc_cache = CACHE_DIR / "documents.npy"

if doc_cache.exists():
    print("\nLoading cached document embeddings...")
    doc_embeddings = np.load(doc_cache)

else:
    print("\nEncoding 5,000 documents...")
    start = time.perf_counter()

    doc_embeddings = model.encode_document(
        doc_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = time.perf_counter() - start

    np.save(doc_cache, doc_embeddings)

    print(
        f"Document encoding completed in "
        f"{elapsed / 60:.2f} minutes."
    )


# --------------------------------------------------
# QUERY EMBEDDINGS
# --------------------------------------------------

query_cache = CACHE_DIR / "queries.npy"

if query_cache.exists():
    print("\nLoading cached query embeddings...")
    query_embeddings = np.load(query_cache)

else:
    print("\nEncoding 1,000 validation queries...")
    start = time.perf_counter()

    query_embeddings = model.encode_query(
        query_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = time.perf_counter() - start

    np.save(query_cache, query_embeddings)

    print(
        f"Query encoding completed in "
        f"{elapsed / 60:.2f} minutes."
    )


print("\nDocument matrix:", doc_embeddings.shape)
print("Query matrix:", query_embeddings.shape)


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

print("\nComputing similarity matrix...")

start = time.perf_counter()

# Embeddings are normalized, so dot product = cosine similarity.
scores = query_embeddings @ doc_embeddings.T

retrieval_time = time.perf_counter() - start

print(
    f"Similarity computation: "
    f"{retrieval_time:.3f} seconds"
)


# --------------------------------------------------
# RANK TOP 100
# --------------------------------------------------

print("\nRanking documents...")

TOP_K = 100

rankings = {}

for i, query_id in enumerate(query_ids):
    row_scores = scores[i]

    candidate_indices = np.argpartition(
        -row_scores,
        TOP_K - 1
    )[:TOP_K]

    candidate_indices = candidate_indices[
        np.argsort(-row_scores[candidate_indices])
    ]

    rankings[query_id] = [
        doc_ids[idx]
        for idx in candidate_indices
    ]


# --------------------------------------------------
# METRICS
# --------------------------------------------------

metrics = evaluate_rankings(
    rankings,
    relevant_docs
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for name, value in metrics.items():
    print(f"{name:12}: {value:.6f}")

print(
    f"\nAverage similarity-search latency/query: "
    f"{retrieval_time / len(queries) * 1000:.3f} ms"
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output = {
    "model": MODEL,
    "max_seq_length": MAX_SEQ_LENGTH,
    "batch_size": BATCH_SIZE,
    "code_preprocessing": "large_literal_compaction",
    "validation_queries": len(queries),
    "corpus_size": len(corpus),
    "metrics": metrics,
    "similarity_search_seconds": retrieval_time,
}

with open(
    RESULTS_DIR / "qwen06b_baseline.json",
    "w"
) as f:
    json.dump(output, f, indent=2)

print(
    "\nSaved results to "
    "results/qwen06b_baseline.json"
)

print("\n✅ First real benchmark completed.")
