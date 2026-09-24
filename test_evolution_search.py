from src.evolution_search import (
    EvolutionSearchEngine,
)


engine = EvolutionSearchEngine()


queries = [
    "show how binary search changed between versions",

    "find the older implementation of factorial",
]


for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)

    response = engine.search(
        query,
        top_k=3
    )

    print(
        "STRATEGY:",
        response["strategy"]
    )

    print("\nRESULTS:")

    for result in response[
        "results"
    ]:
        print(
            f"\nLineage: "
            f"{result['logical_id']}"
        )

        print(
            f"Versions: "
            f"{result['version_count']}"
        )

        for item in result[
            "timeline"
        ]:
            print(
                f"  {item['version']:3} | "
                f"{item['change_type']:22} | "
                f"similarity="
                f"{item['previous_similarity']}"
            )


print(
    "\n✅ Evolutionary search test completed."
)
