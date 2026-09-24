from src.structural_search import (
    StructuralSearchEngine,
)


engine = StructuralSearchEngine()


queries = [
    "which files import math?",
    "find usages of range",
    "find recursive functions",
    "which classes contain nested loops?",
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

    for result in response["results"]:
        print(
            f"\n{result['id']} "
            f"score={result['score']:.2f}"
        )

        for evidence in result[
            "evidence"
        ]:
            print(
                "  ",
                evidence
            )

print(
    "\n✅ Structural search test completed."
)
