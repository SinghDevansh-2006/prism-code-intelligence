from src.structural_search import (
    StructuralSearchEngine,
)


engine = StructuralSearchEngine()


queries = [
    "which classes contain nested loops?",

    "which functions contain nested loops?",

    "which functions call range before print?",
]


for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)

    response = engine.search(
        query,
        top_k=5
    )

    print("ROUTE:", response["route"])
    print("TERMS:", response["terms"])

    print("\nRESULTS:")

    for result in response[
        "results"
    ]:
        print(
            f"\n{result['id']} "
            f"score={result['score']:.2f}"
        )

        if "scope" in result:
            print(
                "  scope:",
                result["scope"]
            )

        if result.get("class"):
            print(
                "  class:",
                result["class"]
            )

        for evidence in result[
            "evidence"
        ]:
            print(
                " ",
                evidence
            )


print(
    "\n✅ Scope-aware structural "
    "search test completed."
)
