from src.agentic_engine import (
    AgenticCodeEngine,
)


engine = AgenticCodeEngine(
    enable_reranker=True,

    version_index_root=(
        "runtime_index/"
        "versioned_semantic_test"
    ),
)


query = (
    "show how binary search "
    "changed between versions"
)


print("=" * 80)
print("AGENTIC EVOLUTION TEST")
print("=" * 80)

response = engine.search(
    query,
    top_k=4,
)


print("\nQUERY:")
print(response["query"])


print("\nPLAN:")
print(
    "Route:",
    response["plan"]["route"]
)

print(
    "Reason:",
    response["plan"]["reason"]
)


print("\nEXECUTION:")

print(
    "Strategy:",
    response["execution"][
        "strategy"
    ]
)

for step in response[
    "execution"
]["steps"]:
    print(
        "  ->",
        step
    )


if "timings_ms" in response[
    "execution"
]:
    print(
        "\nTimings:",
        response["execution"][
            "timings_ms"
        ]
    )


print("\nRESULTS:")

for result in response[
    "results"
]:
    print(
        f"\n#{result['rank']} "
        f"{result['logical_id']} "
        f"{result['version']}"
    )

    print(
        f"score: "
        f"{result['score']:.4f}"
    )

    print(
        f"change: "
        f"{result['change_type']}"
    )

    print("timeline:")

    for item in result[
        "timeline"
    ]:
        print(
            f"  {item['version']} | "
            f"{item['change_type']} | "
            f"similarity="
            f"{item['previous_similarity']}"
        )


print(
    f"\nTotal agent time: "
    f"{response['total_ms']:.2f} ms"
)

print(
    "\n✅ Agentic evolutionary route works."
)
