import sys

from src.retriever import CodeRetriever


if len(sys.argv) < 2:
    print(
        'Usage: python search_cli.py '
        '"natural language query"'
    )
    raise SystemExit(1)


query = " ".join(
    sys.argv[1:]
)


engine = CodeRetriever(
    enable_reranker=True
)


response = engine.search(
    query,
    top_k=5
)


print("\n" + "=" * 80)
print("QUERY")
print("=" * 80)
print(response["query"])


print("\n" + "=" * 80)
print("RETRIEVAL INFO")
print("=" * 80)

info = response["retrieval"]

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

print(
    "\nTimings:"
)

for name, value in info[
    "timings_ms"
].items():
    print(
        f"  {name:24} "
        f"{value:8.2f} ms"
    )


print("\n" + "=" * 80)
print("TOP RESULTS")
print("=" * 80)

for result in response[
    "results"
]:
    print(
        f"\n#{result['rank']} "
        f"{result['id']} "
        f"(dense score="
        f"{result['score']:.4f})"
    )

    if result["title"]:
        print(
            "Title:",
            result["title"]
        )

    print(
        result["code"][:700]
    )

    print("-" * 80)
