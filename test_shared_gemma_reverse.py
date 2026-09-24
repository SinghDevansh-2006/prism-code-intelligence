from src.agentic_engine import AgenticCodeEngine


engine = AgenticCodeEngine(
    enable_reranker=True,
    version_index_root=(
        "runtime_index/"
        "versioned_semantic_test"
    ),
)


print("=" * 80)
print("SHARED EMBEDDINGGEMMA — REVERSE ORDER")
print("=" * 80)


print("\n1. SEMANTIC QUERY")

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


print("\n2. EVOLUTION QUERY")

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
    evolution["results"][0]["logical_id"],
    evolution["results"][0]["version"],
)


print("\n3. OBJECT IDENTITY")

same_model = (
    engine.semantic_engine.gemma
    is engine.version_engine.model
)

print(
    "Same EmbeddingGemma object:",
    same_model
)

assert same_model


print(
    "\n✅ Semantic and evolution retrieval "
    "share one EmbeddingGemma instance "
    "in reverse route order."
)
