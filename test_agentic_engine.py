from src.agentic_engine import (
    AgenticCodeEngine,
)


engine = AgenticCodeEngine(
    enable_reranker=True
)


queries = [
    "check whether a tic tac toe board state is valid",

    "which files import math?",

    "which functions contain nested loops?",

    "which functions call range before print?",
]


for query in queries:
    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    response = engine.search(
        query,
        top_k=3
    )

    print("\nPLAN")
    print("-" * 80)

    print(
        "Route:",
        response["plan"]["route"]
    )

    print(
        "Reason:",
        response["plan"]["reason"]
    )

    print(
        "Terms:",
        response["plan"][
            "extracted_terms"
        ]
    )

    print("\nEXECUTION")
    print("-" * 80)

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

    print(
        f"\nTotal time: "
        f"{response['total_ms']:.2f} ms"
    )

    print("\nRESULTS")
    print("-" * 80)

    for result in response[
        "results"
    ]:
        print(
            f"\n#{result.get('rank', '-')}"
            f" {result['id']}"
        )

        if "score" in result:
            print(
                "Score:",
                result["score"]
            )

        if result.get(
            "evidence"
        ):
            for evidence in result[
                "evidence"
            ]:
                print(
                    " ",
                    evidence
                )

        print(
            result["code"][:250]
            .replace(
                "\n",
                " "
            )
        )


print(
    "\n✅ Unified agentic engine test completed."
)
