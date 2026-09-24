import time

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)

QUERY = (
    "check whether a tic tac toe "
    "board state is valid"
)


print("=" * 80)
print("PERSISTENT API — COLD VS WARM SEMANTIC")
print("=" * 80)


for run in range(1, 4):
    wall_start = time.perf_counter()

    response = client.post(
        "/search",
        json={
            "query": QUERY,
            "top_k": 3,
        },
    )

    wall_ms = (
        time.perf_counter()
        - wall_start
    ) * 1000

    assert response.status_code == 200

    data = response.json()

    timings = data[
        "execution"
    ].get(
        "timings_ms",
        {}
    )

    print(
        f"\nRUN {run}"
    )

    print(
        f"Route:            "
        f"{data['plan']['route']}"
    )

    print(
        f"API wall time:    "
        f"{wall_ms:.2f} ms"
    )

    print(
        f"Agent total time: "
        f"{data['total_ms']:.2f} ms"
    )

    print(
        f"Gemma:            "
        f"{timings.get('gemma_ms', 'n/a')} ms"
    )

    print(
        f"Qwen:             "
        f"{timings.get('qwen_ms', 'n/a')} ms"
    )

    print(
        f"Vector search:    "
        f"{timings.get('vector_ms', 'n/a')} ms"
    )

    print(
        f"Reranker:         "
        f"{data['execution']['reranker_triggered']}"
    )

    print(
        "Top result:       ",
        data["results"][0]["id"],
    )


health = client.get(
    "/health"
).json()


print("\n" + "=" * 80)
print("FINAL HEALTH")
print("=" * 80)

print(
    health["models_loaded"]
)

print(
    "\n✅ Warm semantic API benchmark completed."
)
