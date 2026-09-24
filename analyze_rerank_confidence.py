import numpy as np

from src.data_loader import load_dev_benchmark


queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]

doc_index = {
    did: i
    for i, did in enumerate(doc_ids)
}


# --------------------------------------------------
# LOAD CURRENT BEST DENSE SYSTEM
# --------------------------------------------------

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


# --------------------------------------------------
# CONFIDENCE + TRUE RANK
# --------------------------------------------------

records = []

for i, qid in enumerate(query_ids):
    row = scores[i]

    top2_idx = np.argpartition(
        -row,
        1
    )[:2]

    top2_scores = np.sort(
        row[top2_idx]
    )[::-1]

    margin = float(
        top2_scores[0] - top2_scores[1]
    )

    relevant_idx = doc_index[
        relevant_docs[qid]
    ]

    relevant_score = row[
        relevant_idx
    ]

    rank = int(
        1 + np.sum(row > relevant_score)
    )

    records.append(
        (
            margin,
            rank,
            qid,
        )
    )


# lowest margin = least confident
records.sort(
    key=lambda x: x[0]
)


print("=" * 78)
print("SELECTIVE RERANKING CONFIDENCE ANALYSIS")
print("=" * 78)

total_non_rank1 = sum(
    rank > 1
    for _, rank, _ in records
)

total_top10_misses = sum(
    rank > 10
    for _, rank, _ in records
)

total_top20_recoverable = sum(
    10 < rank <= 20
    for _, rank, _ in records
)

print(f"\nQueries total:                  {len(records)}")
print(f"Not currently rank 1:          {total_non_rank1}")
print(f"Currently outside top 10:      {total_top10_misses}")
print(f"Recoverable from ranks 11-20:  {total_top20_recoverable}")


print("\n" + "=" * 78)
print("LOW-CONFIDENCE GATING")
print("=" * 78)

for fraction in [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]:
    count = int(
        len(records) * fraction
    )

    selected = records[:count]

    captured_non_rank1 = sum(
        rank > 1
        for _, rank, _ in selected
    )

    captured_misses = sum(
        rank > 10
        for _, rank, _ in selected
    )

    captured_recoverable = sum(
        10 < rank <= 20
        for _, rank, _ in selected
    )

    print(
        f"\nLowest-margin {int(fraction * 100):2}% "
        f"({count:3} queries)"
    )

    print(
        f"  captures rank>1:       "
        f"{captured_non_rank1:3}/{total_non_rank1}"
        f" "
        f"({captured_non_rank1 / total_non_rank1:.1%})"
    )

    print(
        f"  captures rank>10:      "
        f"{captured_misses:3}/{total_top10_misses}"
        f" "
        f"({captured_misses / total_top10_misses:.1%})"
    )

    print(
        f"  captures rank 11-20:   "
        f"{captured_recoverable:3}/{total_top20_recoverable}"
        f" "
        f"({captured_recoverable / total_top20_recoverable:.1%})"
    )

    estimated_minutes = (
        count * 20 / 8.16 / 60
    )

    print(
        f"  estimated rerank time: "
        f"{estimated_minutes:.1f} min"
    )


print("\n" + "=" * 78)
print("27 CURRENT TOP-10 MISSES")
print("=" * 78)

for margin, rank, qid in records:
    if rank > 10:
        print(
            f"{qid:8} | "
            f"rank={rank:4} | "
            f"margin={margin:.6f}"
        )


print("\n✅ Confidence analysis completed.")
