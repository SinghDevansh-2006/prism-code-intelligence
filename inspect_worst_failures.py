import json
import numpy as np
from datasets import load_dataset

from src.data_loader import load_dev_benchmark

WORST_N = 5

queries, corpus, relevant_docs = load_dev_benchmark()

query_by_id = {
    row["_id"]: row
    for row in queries
}

doc_by_id = {
    row["_id"]: row
    for row in corpus
}

doc_ids = [row["_id"] for row in corpus]

query_embeddings = np.load(
    "cache/qwen06b/queries.npy"
)

doc_embeddings = np.load(
    "cache/qwen06b/documents.npy"
)

with open(
    "results/qwen06b_failure_analysis.json"
) as f:
    failure_data = json.load(f)

worst = failure_data["worst_20"][:WORST_N]

query_ids_in_order = [
    row["_id"] for row in queries
]

query_index = {
    qid: i
    for i, qid in enumerate(query_ids_in_order)
}

scores = query_embeddings @ doc_embeddings.T

for case in worst:
    qid = case["query_id"]
    correct_doc_id = case["doc_id"]

    q_idx = query_index[qid]
    row_scores = scores[q_idx]

    top_indices = np.argsort(-row_scores)[:3]

    query = query_by_id[qid]
    correct_doc = doc_by_id[correct_doc_id]

    print("\n" + "=" * 100)
    print(
        f"QUERY {qid} | "
        f"CORRECT {correct_doc_id} | "
        f"CORRECT RANK {case['rank']}"
    )
    print("=" * 100)

    print("\nQUERY:")
    print(query["text"][:1800])

    print("\n--- CORRECT CODE ---")
    print(correct_doc["text"][:1200])

    print("\n--- MODEL TOP 3 ---")

    for rank, idx in enumerate(top_indices, 1):
        doc_id = doc_ids[idx]
        doc = doc_by_id[doc_id]

        print(
            f"\n#{rank} {doc_id} "
            f"score={row_scores[idx]:.4f}"
        )

        print(doc["text"][:700])

        print("-" * 70)
