from pathlib import Path
import shutil

from src.versioned_retriever import (
    VersionedSemanticIndex,
)


TEST_DIR = (
    "runtime_index/"
    "versioned_semantic_test"
)

shutil.rmtree(
    TEST_DIR,
    ignore_errors=True,
)


engine = VersionedSemanticIndex(
    root=TEST_DIR
)


versions = [
    (
        "binary_search",
        "v1",
        """
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
""",
    ),

    (
        "binary_search",
        "v2",
        """
def binary_search(arr, target):
    lo = 0
    hi = len(arr) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1
""",
    ),

    (
        "binary_search",
        "v3",
        """
from bisect import bisect_left

def binary_search(arr, target):
    pos = bisect_left(arr, target)

    if pos < len(arr) and arr[pos] == target:
        return pos

    return -1
""",
    ),

    (
        "factorial",
        "v1",
        """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
    ),
]


print("=" * 80)
print("INCREMENTAL SEMANTIC VERSION INDEX")
print("=" * 80)


for logical_id, version, code in versions:
    result = engine.add_version(
        logical_id=logical_id,
        version=version,
        code=code,
    )

    print(
        f"\n{logical_id} {version}"
    )

    print(
        f"  change:       "
        f"{result['change_type']}"
    )

    print(
        f"  lineage:      "
        f"{result['lineage_index_ms']:.3f} ms"
    )

    print(
        f"  embedding:    "
        f"{result['embedding_ms']:.3f} ms"
    )

    print(
        f"  persistence:  "
        f"{result['persistence_ms']:.3f} ms"
    )

    print(
        f"  TOTAL UPDATE: "
        f"{result['total_update_ms']:.3f} ms"
    )


print("\n" + "=" * 80)
print("SEARCH: iterative binary search")
print("=" * 80)

response = engine.search(
    "iterative binary search over a sorted array",
    top_k=4,
)

for result in response["results"]:
    print(
        f"#{result['rank']} "
        f"{result['logical_id']} "
        f"{result['version']} | "
        f"{result['change_type']} | "
        f"score={result['score']:.4f}"
    )

print(
    "\nSearch timings:",
    response["timings_ms"]
)


print("\n" + "=" * 80)
print("SEARCH: recursive factorial")
print("=" * 80)

response = engine.search(
    "recursive factorial implementation",
    top_k=4,
)

for result in response["results"]:
    print(
        f"#{result['rank']} "
        f"{result['logical_id']} "
        f"{result['version']} | "
        f"{result['change_type']} | "
        f"score={result['score']:.4f}"
    )

print(
    "\nSearch timings:",
    response["timings_ms"]
)

print(
    "\n✅ Incremental semantic retrieval "
    "across versions works."
)
