from src.query_router import route_query


queries = [
    "check whether a tic tac toe board state is valid",

    "find code that performs binary search",

    "which functions call parse_config before save_config?",

    "find usages of torch.nn.Linear",

    "which files import requests?",

    "find recursive functions that traverse a tree",

    "show how this implementation changed between versions",

    "find the older implementation of authentication",

    "which classes contain nested loops?",

    "code that sorts a string according to a custom ordering",
]


print("=" * 80)
print("AGENTIC QUERY ROUTER")
print("=" * 80)


for query in queries:
    plan = route_query(
        query
    )

    print("\nQUERY:")
    print(query)

    print(
        f"ROUTE:      {plan.route}"
    )

    print(
        f"CONFIDENCE: {plan.confidence:.2f}"
    )

    print(
        f"TERMS:      {plan.extracted_terms}"
    )

    print(
        f"WHY:        {plan.reason}"
    )

    print("-" * 80)


print(
    "\n✅ Query router test completed."
)
