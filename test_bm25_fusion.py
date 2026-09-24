import numpy as np

from src.data_loader import load_dev_benchmark
from src.metrics import evaluate_rankings


queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]

doc_index = {
    doc_id: i
    for i, doc_id in enumerate(doc_ids)
}


# --------------------------------------------------
# LOAD CACHED SCORES
# --------------------------------------------------

qwen_q = np.load("cache/qwen06b/queries.npy")
qwen_d = np.load("cache/qwen06b/documents.npy")

gemma_q = np.load("cache/embeddinggemma/queries.npy")
gemma_d = np.load("cache/embeddinggemma/documents.npy")

bm25_scores = np.load("cache/bm25/scores.npy")


# --------------------------------------------------
# CURRENT BEST DENSE FUSION
# --------------------------------------------------

qwen_scores = qwen_q @ qwen_d.T
gemma_scores = gemma_q @ gemma_d.T


def row_zscore(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


qwen_scores = row_zscore(qwen_scores)
gemma_scores = row_zscore(gemma_scores)

dense_scores = (
    0.65 * gemma_scores
    + 0.35 * qwen_scores
)


# --------------------------------------------------
# RANK MATRICES
# rank position starts at 1
# --------------------------------------------------

print("=" * 78)
print("BM25 COMPLEMENTARITY + RRF FUSION")
print("=" * 78)

print("\nBuilding rank matrices...")

dense_order = np.argsort(
    -dense_scores,
    axis=1
)

bm25_order = np.argsort(
    -bm25_scores,
    axis=1
)

dense_ranks = np.empty_like(
    dense_order,
    dtype=np.int32
)

bm25_ranks = np.empty_like(
    bm25_order,
    dtype=np.int32
)

positions = np.arange(
    1,
    len(doc_ids) + 1,
    dtype=np.int32
)

for i in range(len(queries)):
    dense_ranks[i, dense_order[i]] = positions
    bm25_ranks[i, bm25_order[i]] = positions


# --------------------------------------------------
# COMPLEMENTARITY ANALYSIS
# --------------------------------------------------

dense_misses_100 = []
bm25_rescues_100 = []
bm25_rescues_500 = []

for i, qid in enumerate(query_ids):
    relevant_id = relevant_docs[qid]
    idx = doc_index[relevant_id]

    dr = int(dense_ranks[i, idx])
    br = int(bm25_ranks[i, idx])

    if dr > 100:
        dense_misses_100.append(
            (qid, relevant_id, dr, br)
        )

        if br <= 100:
            bm25_rescues_100.append(
                (qid, relevant_id, dr, br)
            )

        if br <= 500:
            bm25_rescues_500.append(
                (qid, relevant_id, dr, br)
            )


print("\nDENSE TOP-100 MISSES")
print("-" * 78)

print(
    f"Dense relevant doc outside top 100: "
    f"{len(dense_misses_100)}"
)

print(
    f"Recovered by BM25 top 100: "
    f"{len(bm25_rescues_100)}"
)

print(
    f"Recovered by BM25 top 500: "
    f"{len(bm25_rescues_500)}"
)

if dense_misses_100:
    print("\nMiss details:")
    for qid, did, dr, br in dense_misses_100:
        print(
            f"{qid:8} -> {did:8} | "
            f"dense rank={dr:4} | "
            f"BM25 rank={br:4}"
        )


# --------------------------------------------------
# WEIGHTED RECIPROCAL RANK FUSION
# --------------------------------------------------

RRF_K = 60

bm25_weights = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
]

results = []

print("\n" + "=" * 78)
print("WEIGHTED RRF RESULTS")
print("Dense weight = 1.0")
print(f"RRF k = {RRF_K}")
print("=" * 78)

dense_rrf = 1.0 / (
    RRF_K + dense_ranks
)

bm25_rrf_base = 1.0 / (
    RRF_K + bm25_ranks
)

for beta in bm25_weights:
    fused = (
        dense_rrf
        + beta * bm25_rrf_base
    )

    rankings = {}

    for i, qid in enumerate(query_ids):
        indices = np.argpartition(
            -fused[i],
            99
        )[:100]

        indices = indices[
            np.argsort(-fused[i][indices])
        ]

        rankings[qid] = [
            doc_ids[idx]
            for idx in indices
        ]

    metrics = evaluate_rankings(
        rankings,
        relevant_docs
    )

    results.append(
        (
            beta,
            metrics["ndcg@10"],
            metrics["mrr"],
            metrics["recall@10"],
            metrics["recall@100"],
        )
    )

    print(
        f"BM25 weight={beta:0.2f} | "
        f"NDCG@10={metrics['ndcg@10']:.6f} | "
        f"MRR={metrics['mrr']:.6f} | "
        f"R@10={metrics['recall@10']:.6f} | "
        f"R@100={metrics['recall@100']:.6f}"
    )


best = max(
    results,
    key=lambda x: (x[1], x[2])
)

print("\n" + "=" * 78)
print("BEST RRF RESULT")
print("=" * 78)

print(f"BM25 weight: {best[0]:.2f}")
print(f"NDCG@10:     {best[1]:.6f}")
print(f"MRR:         {best[2]:.6f}")
print(f"Recall@10:   {best[3]:.6f}")
print(f"Recall@100:  {best[4]:.6f}")

print("\nCurrent dense reference:")
print("NDCG@10:     0.917734")
print("MRR:         0.900386")
print("Recall@10:   0.973000")
print("Recall@100:  0.993000")

print("\n✅ BM25 fusion analysis completed.")
