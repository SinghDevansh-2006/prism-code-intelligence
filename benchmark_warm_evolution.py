import time

from src.agentic_engine import AgenticCodeEngine


engine = AgenticCodeEngine(
    enable_reranker=True,
    version_index_root=(
        "runtime_index/"
        "versioned_semantic_test"
    ),
)


queries = [
    "show how binary search changed between versions",
    "find the older implementation of factorial",
    "show binary search version history",
]


print("=" * 80)
print("COLD VS WARM EVOLUTION LATENCY")
print("=" * 80)


for i, query in enumerate(
    queries,
    start=1
):
    start = time.perf_counter()

    response = engine.search(
        query,
        top_k=4
    )

    wall_ms = (
        time.perf_counter()
        - start
    ) * 1000

    retrieval_ms = (
        response["execution"]
        .get(
            "timings_ms",
            {}
        )
        .get(
            "total",
            0.0
        )
    )

    print(
        f"\nQuery {i}: {query}"
    )

    print(
        f"Route:          "
        f"{response['plan']['route']}"
    )

    print(
        f"Agent wall time: "
        f"{wall_ms:.2f} ms"
    )

    print(
        f"Retrieval time:  "
        f"{retrieval_ms:.2f} ms"
    )

    if response["results"]:
        first = response[
            "results"
        ][0]

        print(
            f"Top result:      "
            f"{first['logical_id']} "
            f"{first['version']}"
        )


print(
    "\n✅ Warm evolution benchmark completed."
)
