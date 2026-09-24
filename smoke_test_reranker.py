from src.runtime_device import get_device

import time

import numpy as np
from sentence_transformers import CrossEncoder

from src.data_loader import load_dev_benchmark
from src.code_normalizer import compact_large_literals


MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
TOP_K = 20
NUM_QUERIES = 50
BATCH_SIZE = 16


print("=" * 72)
print("RERANKER SMOKE + SPEED TEST")
print("=" * 72)

print("\nLoading benchmark...")
queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]

doc_lookup = {
    row["_id"]: compact_large_literals(row["text"])
    for row in corpus
}

query_lookup = {
    row["_id"]: row["text"]
    for row in queries
}


# --------------------------------------------------
# CURRENT DENSE SCORES
# --------------------------------------------------

print("Loading cached embeddings...")

qwen_q = np.load("cache/qwen06b/queries.npy")
qwen_d = np.load("cache/qwen06b/documents.npy")

gemma_q = np.load("cache/embeddinggemma/queries.npy")
gemma_d = np.load("cache/embeddinggemma/documents.npy")


def row_zscore(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


qwen_scores = row_zscore(
    qwen_q @ qwen_d.T
)

gemma_scores = row_zscore(
    gemma_q @ gemma_d.T
)

dense_scores = (
    0.65 * gemma_scores
    + 0.35 * qwen_scores
)


# --------------------------------------------------
# BUILD 50 × TOP-20 PAIRS = 1,000 PAIRS
# --------------------------------------------------

pairs = []

for i in range(NUM_QUERIES):
    scores = dense_scores[i]

    top_indices = np.argpartition(
        -scores,
        TOP_K - 1
    )[:TOP_K]

    top_indices = top_indices[
        np.argsort(-scores[top_indices])
    ]

    qid = query_ids[i]
    query_text = query_lookup[qid]

    for idx in top_indices:
        did = doc_ids[idx]

        pairs.append(
            (
                query_text,
                doc_lookup[did]
            )
        )


print(
    f"\nPrepared {len(pairs)} query-code pairs."
)


# --------------------------------------------------
# LOAD RERANKER
# --------------------------------------------------

print("\nLoading reranker...")

model = CrossEncoder(
    MODEL,
    device=get_device(),
    max_length=512
)


# --------------------------------------------------
# SPEED TEST
# --------------------------------------------------

print("\nScoring pairs...")

start = time.perf_counter()

scores = model.predict(
    pairs,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)

elapsed = time.perf_counter() - start

pairs_per_second = len(pairs) / elapsed

estimated_full_seconds = (
    1000 * TOP_K
) / pairs_per_second


print("\n" + "=" * 72)
print("RERANKER SPEED RESULTS")
print("=" * 72)

print(f"Model:                 {MODEL}")
print(f"Device:                {get_device()}")
print(f"Max length:            512")
print(f"Pairs tested:          {len(pairs)}")
print(f"Elapsed:               {elapsed:.2f} sec")
print(f"Pairs/sec:             {pairs_per_second:.2f}")

print(
    f"Estimated 1,000-query top-{TOP_K} "
    f"rerank time: "
    f"{estimated_full_seconds / 60:.2f} min"
)

print("\nScore sample:")
print(scores[:10])

print(
    "\n✅ Reranker smoke test completed."
)
