import json
import numpy as np

from src.data_loader import load_dev_benchmark

queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]

doc_index = {
    doc_id: i
    for i, doc_id in enumerate(doc_ids)
}

query_embeddings = np.load(
    "cache/qwen06b/queries.npy"
)

doc_embeddings = np.load(
    "cache/qwen06b/documents.npy"
)

scores = query_embeddings @ doc_embeddings.T

results = []

for i, query_id in enumerate(query_ids):
    correct_doc = relevant_docs[query_id]
    correct_idx = doc_index[correct_doc]

    correct_score = scores[i, correct_idx]

    # Exact rank without sorting all 5000 documents
    rank = int(
        np.sum(scores[i] > correct_score) + 1
    )

    results.append({
        "query_id": query_id,
        "doc_id": correct_doc,
        "rank": rank,
        "score": float(correct_score),
    })

ranks = np.array([x["rank"] for x in results])

print("=" * 70)
print("EXACT RANK ANALYSIS")
print("=" * 70)

print("Rank 1:", int((ranks == 1).sum()))
print("Rank 2-5:", int(((ranks >= 2) & (ranks <= 5)).sum()))
print("Rank 6-10:", int(((ranks >= 6) & (ranks <= 10)).sum()))
print("Rank 11-20:", int(((ranks >= 11) & (ranks <= 20)).sum()))
print("Rank 21-100:", int(((ranks >= 21) & (ranks <= 100)).sum()))
print("Rank >100:", int((ranks > 100).sum()))

print("\nMedian rank:", float(np.median(ranks)))
print("Mean rank:", float(np.mean(ranks)))
print("Worst rank:", int(ranks.max()))

worst = sorted(
    results,
    key=lambda x: x["rank"],
    reverse=True
)

print("\n" + "=" * 70)
print("20 WORST QUERIES")
print("=" * 70)

for item in worst[:20]:
    print(
        f'{item["query_id"]:8} | '
        f'{item["doc_id"]:8} | '
        f'rank={item["rank"]:4} | '
        f'score={item["score"]:.4f}'
    )

with open(
    "results/qwen06b_failure_analysis.json",
    "w"
) as f:
    json.dump(
        {
            "all_results": results,
            "worst_20": worst[:20],
        },
        f,
        indent=2,
    )

print(
    "\nSaved to "
    "results/qwen06b_failure_analysis.json"
)

print("\n✅ Failure analysis completed.")
