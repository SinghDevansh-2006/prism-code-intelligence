import numpy as np

from src.data_loader import load_dev_benchmark


queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]

doc_index = {
    doc_id: i
    for i, doc_id in enumerate(doc_ids)
}


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

scores = (
    0.65 * gemma_scores
    + 0.35 * qwen_scores
)


ranks = []
details = []

for i, qid in enumerate(query_ids):
    relevant_id = relevant_docs[qid]
    relevant_idx = doc_index[relevant_id]

    score = scores[i]

    relevant_score = score[relevant_idx]

    rank = int(
        1 + np.sum(score > relevant_score)
    )

    ranks.append(rank)

    if rank > 10:
        details.append(
            (
                rank,
                qid,
                relevant_id,
            )
        )


ranks = np.array(ranks)

print("=" * 74)
print("DENSE FUSION RANK ANALYSIS")
print("=" * 74)

print(f"Rank 1:       {(ranks == 1).sum()}")
print(f"Rank 2-3:     {((ranks >= 2) & (ranks <= 3)).sum()}")
print(f"Rank 4-5:     {((ranks >= 4) & (ranks <= 5)).sum()}")
print(f"Rank 6-10:    {((ranks >= 6) & (ranks <= 10)).sum()}")
print(f"Rank 11-20:   {((ranks >= 11) & (ranks <= 20)).sum()}")
print(f"Rank 21-50:   {((ranks >= 21) & (ranks <= 50)).sum()}")
print(f"Rank 51-100:  {((ranks >= 51) & (ranks <= 100)).sum()}")
print(f"Rank >100:    {(ranks > 100).sum()}")

print("\nRecall by candidate depth:")

for k in [5, 10, 15, 20, 30, 50, 100]:
    recall = np.mean(ranks <= k)

    print(
        f"Recall@{k:<3}: {recall:.6f}"
    )


print("\nReciprocal-rank statistics:")
print(f"Median rank: {np.median(ranks):.1f}")
print(f"Mean rank:   {np.mean(ranks):.3f}")
print(f"Worst rank:  {np.max(ranks)}")


print("\n" + "=" * 74)
print("TOP-10 MISSES, SORTED BY RANK")
print("=" * 74)

details.sort()

for rank, qid, did in details:
    if rank > 100:
        continue

    print(
        f"{qid:8} -> {did:8} | rank={rank}"
    )


print("\n" + "=" * 74)
print("TOP-100 MISSES")
print("=" * 74)

for rank, qid, did in sorted(
    details,
    reverse=True
):
    if rank > 100:
        print(
            f"{qid:8} -> {did:8} | rank={rank}"
        )

print("\n✅ Dense rank analysis completed.")
