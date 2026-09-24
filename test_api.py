from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


print("=" * 80)
print("PERSISTENT API SMOKE TEST")
print("=" * 80)


# ------------------------------------------------------------------
# HEALTH BEFORE ANY MODEL IS USED
# ------------------------------------------------------------------

health = client.get(
    "/health"
)

assert health.status_code == 200

print("\nHEALTH — BEFORE SEARCH")
print(health.json())


# ------------------------------------------------------------------
# STRUCTURAL QUERY
# ------------------------------------------------------------------

response = client.post(
    "/search",
    json={
        "query":
            "which functions call range before print?",
        "top_k": 3,
    },
)

assert response.status_code == 200

data = response.json()

print("\n" + "=" * 80)
print("STRUCTURAL REQUEST")
print("=" * 80)

print(
    "Route:",
    data["plan"]["route"]
)

print(
    "Strategy:",
    data["execution"]["strategy"]
)

print(
    "Total ms:",
    round(
        data["total_ms"],
        2
    )
)

print("Results:")

for result in data["results"]:
    print(
        " ",
        result["id"],
        result.get("evidence", [])
    )


# ------------------------------------------------------------------
# SEMANTIC QUERY
# ------------------------------------------------------------------

response = client.post(
    "/search",
    json={
        "query":
            "check whether a tic tac toe board state is valid",
        "top_k": 3,
    },
)

assert response.status_code == 200

data = response.json()

print("\n" + "=" * 80)
print("SEMANTIC REQUEST")
print("=" * 80)

print(
    "Route:",
    data["plan"]["route"]
)

print(
    "Strategy:",
    data["execution"]["strategy"]
)

print(
    "Total ms:",
    round(
        data["total_ms"],
        2
    )
)

print(
    "Reranker:",
    data["execution"][
        "reranker_triggered"
    ]
)

print("Results:")

for result in data["results"]:
    print(
        f"  #{result['rank']} "
        f"{result['id']}"
    )


# ------------------------------------------------------------------
# HEALTH AFTER ROUTES HAVE LOADED
# ------------------------------------------------------------------

health = client.get(
    "/health"
)

print("\n" + "=" * 80)
print("HEALTH — AFTER SEARCH")
print("=" * 80)

print(
    health.json()
)


print(
    "\n✅ Persistent API smoke test completed."
)
