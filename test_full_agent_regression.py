from src.agentic_engine import AgenticCodeEngine


engine = AgenticCodeEngine(
    enable_reranker=True,
    version_index_root=(
        "runtime_index/"
        "versioned_semantic_test"
    ),
)


tests = [
    {
        "name": "SEMANTIC",
        "query":
            "check whether a tic tac toe board state is valid",
        "expected_route": "semantic",
    },
    {
        "name": "EXACT USAGE",
        "query":
            "which files import math?",
        "expected_route": "exact_usage",
    },
    {
        "name": "STRUCTURAL",
        "query":
            "which functions call range before print?",
        "expected_route": "structural",
    },
    {
        "name": "EVOLUTION",
        "query":
            "show binary search version history",
        "expected_route": "evolution",
    },
]


print("=" * 80)
print("FULL AGENTIC BACKEND REGRESSION")
print("=" * 80)


for test in tests:
    print("\n" + "=" * 80)
    print(test["name"])
    print("=" * 80)

    response = engine.search(
        test["query"],
        top_k=3,
    )

    route = response[
        "plan"
    ][
        "route"
    ]

    print(
        "Query:",
        test["query"]
    )

    print(
        "Route:",
        route
    )

    print(
        "Strategy:",
        response[
            "execution"
        ][
            "strategy"
        ]
    )

    print(
        f"Total time: "
        f"{response['total_ms']:.2f} ms"
    )

    assert (
        route
        == test["expected_route"]
    ), (
        f"Expected route "
        f"{test['expected_route']}, "
        f"got {route}"
    )

    assert response["results"], (
        f"No results returned for "
        f"{test['name']}"
    )

    first = response[
        "results"
    ][0]

    if route == "evolution":
        print(
            "Top result:",
            first["logical_id"],
            first["version"],
        )

    else:
        print(
            "Top result:",
            first["id"]
        )


print("\n" + "=" * 80)
print("MODEL SHARING CHECK")
print("=" * 80)

assert (
    engine.semantic_engine
    is not None
)

assert (
    engine.structural_engine
    is not None
)

assert (
    engine.version_engine
    is not None
)

same_gemma = (
    engine.semantic_engine.gemma
    is engine.version_engine.model
)

print(
    "Semantic loaded:",
    True
)

print(
    "Structural loaded:",
    True
)

print(
    "Evolution loaded:",
    True
)

print(
    "Shared EmbeddingGemma:",
    same_gemma
)

assert same_gemma


print("\n" + "=" * 80)
print("KEY CORRECTNESS CHECKS")
print("=" * 80)


semantic = engine.search(
    "check whether a tic tac toe board state is valid",
    top_k=1,
)

assert (
    semantic["results"][0]["id"]
    == "d1896"
)

print(
    "Tic-tac-toe semantic result:",
    semantic["results"][0]["id"],
    "✅"
)


structural = engine.search(
    "which functions call range before print?",
    top_k=1,
)

evidence = structural[
    "results"
][0][
    "evidence"
]

assert (
    evidence[0][
        "function"
    ]
    == evidence[1][
        "function"
    ]
)

assert (
    evidence[0]["line"]
    < evidence[1]["line"]
)

print(
    "Scope-aware call ordering:",
    evidence[0]["function"],
    evidence[0]["line"],
    "->",
    evidence[1]["line"],
    "✅"
)


evolution = engine.search(
    "show binary search version history",
    top_k=1,
)

timeline = evolution[
    "results"
][0][
    "timeline"
]

assert len(timeline) == 3

assert (
    timeline[1][
        "change_type"
    ]
    == "near_duplicate_update"
)

assert (
    timeline[2][
        "change_type"
    ]
    == "substantial_update"
)

print(
    "Evolution timeline:",
    [
        (
            item["version"],
            item["change_type"],
        )
        for item in timeline
    ],
    "✅"
)


print(
    "\n✅ All four agentic retrieval routes "
    "passed regression."
)
