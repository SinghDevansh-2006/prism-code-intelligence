import numpy as np

from src.data_loader import load_dev_benchmark
from src.retriever import CodeRetriever


TARGET_QID = "q1595"

queries, _, relevant_docs = load_dev_benchmark()

query_lookup = {
    row["_id"]: row["text"]
    for row in queries
}

query = query_lookup[TARGET_QID]
expected_doc = relevant_docs[TARGET_QID]


engine = CodeRetriever(
    enable_reranker=True
)


# --------------------------------------------------
# DENSE RANK
# --------------------------------------------------

dense_scores, _ = engine._dense_search(
    query
)

dense_order = np.argsort(
    -dense_scores
)

expected_idx = engine.doc_ids.index(
    expected_doc
)

dense_rank = int(
    np.where(
        dense_order == expected_idx
    )[0][0]
    + 1
)


# --------------------------------------------------
# RERANKED ORDER
# --------------------------------------------------

final_order, rerank_ms = engine._rerank(
    query,
    dense_scores,
    dense_order
)

final_rank = int(
    np.where(
        final_order == expected_idx
    )[0][0]
    + 1
)


print("=" * 72)
print("RUNTIME RANK DIAGNOSTIC")
print("=" * 72)

print("Query ID:              ", TARGET_QID)
print("Expected document:     ", expected_doc)
print("Dense rank:            ", dense_rank)
print("Final reranked rank:   ", final_rank)
print(f"Reranker inference:     {rerank_ms:.2f} ms")

print("\nTop 20 after reranking:")

for rank, idx in enumerate(
    final_order[:20],
    start=1
):
    marker = ""

    if engine.doc_ids[idx] == expected_doc:
        marker = "  <-- EXPECTED"

    print(
        f"#{rank:2} "
        f"{engine.doc_ids[idx]:8}"
        f"{marker}"
    )

print("\n✅ Runtime rank diagnostic completed.")
