import numpy as np

from src.data_loader import load_dev_benchmark
from src.metrics import evaluate_rankings

queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]

qwen_q = np.load("cache/qwen06b/queries.npy")
qwen_d = np.load("cache/qwen06b/documents.npy")

gemma_q = np.load("cache/embeddinggemma/queries.npy")
gemma_d = np.load("cache/embeddinggemma/documents.npy")

qwen_scores = qwen_q @ qwen_d.T
gemma_scores = gemma_q @ gemma_d.T


def row_zscore(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


qwen_scores = row_zscore(qwen_scores)
gemma_scores = row_zscore(gemma_scores)

weights = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

results = []

print("=" * 78)
print("FINE-GRAINED DENSE FUSION")
print("alpha = EmbeddingGemma weight")
print("=" * 78)

for alpha in weights:
    fused = (
        alpha * gemma_scores
        + (1.0 - alpha) * qwen_scores
    )

    rankings = {}

    for i, qid in enumerate(query_ids):
        row = fused[i]

        indices = np.argpartition(
            -row,
            99
        )[:100]

        indices = indices[
            np.argsort(-row[indices])
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
            alpha,
            metrics["ndcg@10"],
            metrics["mrr"],
            metrics["recall@10"],
            metrics["recall@100"],
        )
    )

    print(
        f"alpha={alpha:.2f} | "
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
print("BEST")
print("=" * 78)

print(f"Gemma weight: {best[0]:.2f}")
print(f"Qwen weight:  {1.0 - best[0]:.2f}")
print(f"NDCG@10:      {best[1]:.6f}")
print(f"MRR:          {best[2]:.6f}")
print(f"Recall@10:    {best[3]:.6f}")
print(f"Recall@100:   {best[4]:.6f}")

print("\n✅ Fine fusion search completed.")
