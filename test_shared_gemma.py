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


print("=" * 80)
print("SHARED EMBEDDINGGEMMA TEST")
print("=" * 80)


print("\n1. EVOLUTION QUERY")

evolution = engine.search(
    "show binary search version history",
    top_k=3,
)

print(
    "Route:",
    evolution["plan"]["route"]
)

print(
    "Top:",
    evolution["results"][0][
        "logical_id"
    ],
    evolution["results"][0][
        "version"
    ],
)


print("\n2. SEMANTIC QUERY")

semantic = engine.search(
    "check whether a tic tac toe board state is valid",
    top_k=3,
)

print(
    "Route:",
    semantic["plan"]["route"]
)

print(
    "Top:",
    semantic["results"][0]["id"]
)


print("\n3. OBJECT IDENTITY")

same_model = (
    engine.version_engine.model
    is engine.semantic_engine.gemma
)

print(
    "Same EmbeddingGemma object:",
    same_model
)

assert same_model

print(
    "\n✅ Evolution and semantic retrieval "
    "share one EmbeddingGemma instance."
)
