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


print("=" * 80)
print("LIVE LOW-CONFIDENCE RUNTIME TEST")
print("=" * 80)

print("\nQuery ID:", TARGET_QID)
print("Expected relevant doc:", expected_doc)

print("\nQuery:")
print(query)


engine = CodeRetriever(
    enable_reranker=True
)

response = engine.search(
    query,
    top_k=10
)


info = response["retrieval"]

print("\n" + "=" * 80)
print("RETRIEVAL INFO")
print("=" * 80)

print(
    f"Confidence margin: "
    f"{info['confidence_margin']:.6f}"
)

print(
    f"Gate threshold:    "
    f"{info['confidence_threshold']:.6f}"
)

print(
    f"Reranker used:     "
    f"{info['reranker_triggered']}"
)

print("\nTimings:")

for name, value in info["timings_ms"].items():
    print(
        f"  {name:24} "
        f"{value:8.2f} ms"
    )


print("\n" + "=" * 80)
print("TOP 10")
print("=" * 80)

found_rank = None

for result in response["results"]:
    marker = ""

    if result["id"] == expected_doc:
        marker = "  <-- EXPECTED"
        found_rank = result["rank"]

    print(
        f"#{result['rank']:2} "
        f"{result['id']:8} "
        f"score={result['score']:.4f}"
        f"{marker}"
    )


print("\nExpected document final rank:", found_rank)

print(
    "\n✅ Live confidence-gated "
    "runtime test completed."
)
