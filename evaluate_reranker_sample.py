from src.runtime_device import get_device

import time
import numpy as np
from sentence_transformers import CrossEncoder

from src.data_loader import load_dev_benchmark
from src.code_normalizer import compact_large_literals
from src.metrics import evaluate_rankings


MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
TOP_K = 20
SAMPLE_SIZE = 100
BATCH_SIZE = 16
SEED = 42


print("=" * 76)
print("RERANKER QUALITY GATE")
print("=" * 76)

queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]

query_texts = {
    row["_id"]: row["text"]
    for row in queries
}

doc_texts = {
    row["_id"]: compact_large_literals(row["text"])
    for row in corpus
}


# --------------------------------------------------
# DENSE SCORES
# --------------------------------------------------

print("\nLoading cached dense embeddings...")

qwen_q = np.load("cache/qwen06b/queries.npy")
qwen_d = np.load("cache/qwen06b/documents.npy")

gemma_q = np.load("cache/embeddinggemma/queries.npy")
gemma_d = np.load("cache/embeddinggemma/documents.npy")


def row_zscore(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


qwen_scores = row_zscore(qwen_q @ qwen_d.T)
gemma_scores = row_zscore(gemma_q @ gemma_d.T)

dense_scores = (
    0.65 * gemma_scores
    + 0.35 * qwen_scores
)


# --------------------------------------------------
# RANDOM REPRESENTATIVE SAMPLE
# --------------------------------------------------

rng = np.random.default_rng(SEED)

sample_indices = np.sort(
    rng.choice(
        len(queries),
        size=SAMPLE_SIZE,
        replace=False
    )
)

sample_qids = [
    query_ids[i]
    for i in sample_indices
]

sample_relevant = {
    qid: relevant_docs[qid]
    for qid in sample_qids
}


# --------------------------------------------------
# BASELINE RANKINGS + RERANK PAIRS
# --------------------------------------------------

dense_orders = {}
top20_indices = {}
pairs = []

for query_idx in sample_indices:
    qid = query_ids[query_idx]
    row = dense_scores[query_idx]

    order = np.argsort(-row)

    dense_orders[qid] = order
    top20_indices[qid] = order[:TOP_K]

    for doc_idx in order[:TOP_K]:
        did = doc_ids[doc_idx]

        pairs.append(
            (
                query_texts[qid],
                doc_texts[did]
            )
        )


print(
    f"\nPrepared {len(pairs)} pairs "
    f"from {SAMPLE_SIZE} random queries."
)


# --------------------------------------------------
# RERANKER
# --------------------------------------------------

print("\nLoading reranker...")

model = CrossEncoder(
    MODEL,
    device=get_device(),
    max_length=512
)

print("Scoring...")

start = time.perf_counter()

reranker_scores = model.predict(
    pairs,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)

elapsed = time.perf_counter() - start

reranker_scores = np.asarray(
    reranker_scores
).reshape(
    SAMPLE_SIZE,
    TOP_K
)

print(
    f"\nScoring time: {elapsed:.2f} sec "
    f"({elapsed / SAMPLE_SIZE:.3f} sec/query)"
)


# --------------------------------------------------
# DENSE BASELINE ON SAME SAMPLE
# --------------------------------------------------

dense_rankings = {}

for qid in sample_qids:
    order = dense_orders[qid]

    dense_rankings[qid] = [
        doc_ids[idx]
        for idx in order
    ]

dense_metrics = evaluate_rankings(
    dense_rankings,
    sample_relevant
)


print("\n" + "=" * 76)
print("DENSE BASELINE — SAME 100 QUERIES")
print("=" * 76)

for name, value in dense_metrics.items():
    print(f"{name:12}: {value:.6f}")


# --------------------------------------------------
# BLEND DENSE + RERANKER WITHIN TOP 20
# --------------------------------------------------

betas = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

results = []


def zscore_1d(x):
    std = x.std()

    if std < 1e-8:
        return np.zeros_like(x)

    return (
        x - x.mean()
    ) / std


print("\n" + "=" * 76)
print("TOP-20 RERANK RESULTS")
print("beta = reranker weight")
print("=" * 76)


for beta in betas:
    rankings = {}

    for sample_pos, query_idx in enumerate(sample_indices):
        qid = query_ids[query_idx]

        full_dense_order = dense_orders[qid]
        candidate_indices = top20_indices[qid]

        dense_top_scores = dense_scores[
            query_idx,
            candidate_indices
        ]

        rerank_top_scores = reranker_scores[
            sample_pos
        ]

        dense_z = zscore_1d(
            dense_top_scores
        )

        rerank_z = zscore_1d(
            rerank_top_scores
        )

        blended = (
            (1.0 - beta) * dense_z
            + beta * rerank_z
        )

        reranked_positions = np.argsort(
            -blended
        )

        new_top20 = candidate_indices[
            reranked_positions
        ]

        full_order = np.concatenate(
            [
                new_top20,
                full_dense_order[TOP_K:]
            ]
        )

        rankings[qid] = [
            doc_ids[idx]
            for idx in full_order
        ]

    metrics = evaluate_rankings(
        rankings,
        sample_relevant
    )

    results.append(
        (
            beta,
            metrics["ndcg@10"],
            metrics["mrr"],
            metrics["recall@10"],
            metrics["recall@20"],
        )
    )

    print(
        f"beta={beta:.2f} | "
        f"NDCG@10={metrics['ndcg@10']:.6f} | "
        f"MRR={metrics['mrr']:.6f} | "
        f"R@10={metrics['recall@10']:.6f} | "
        f"R@20={metrics['recall@20']:.6f}"
    )


best = max(
    results,
    key=lambda x: (x[1], x[2])
)

print("\n" + "=" * 76)
print("BEST SAMPLE RESULT")
print("=" * 76)

print(f"Reranker weight: {best[0]:.2f}")
print(f"NDCG@10:         {best[1]:.6f}")
print(f"MRR:             {best[2]:.6f}")
print(f"Recall@10:       {best[3]:.6f}")
print(f"Recall@20:       {best[4]:.6f}")

print("\nDense baseline on same sample:")
print(f"NDCG@10:         {dense_metrics['ndcg@10']:.6f}")
print(f"MRR:             {dense_metrics['mrr']:.6f}")
print(f"Recall@10:       {dense_metrics['recall@10']:.6f}")

print("\n✅ Reranker quality gate completed.")
