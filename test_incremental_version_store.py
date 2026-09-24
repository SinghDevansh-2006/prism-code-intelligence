from pathlib import Path

from src.incremental_version_store import (
    IncrementalVersionStore,
)


TEST_PATH = (
    "runtime_index/"
    "version_store_test.json"
)

Path(TEST_PATH).unlink(
    missing_ok=True
)


store = IncrementalVersionStore(
    path=TEST_PATH,
    similarity_threshold=0.75,
)


versions = [
    (
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
]


print("=" * 80)
print("INCREMENTAL VERSION STORE TEST")
print("=" * 80)


for version, code in versions:
    result = store.add_version(
        logical_id="binary_search",
        version=version,
        code=code,
    )

    record = result["record"]

    print(
        f"\n{version}"
    )

    print(
        f"change type:       "
        f"{record['change_type']}"
    )

    print(
        f"similarity:        "
        f"{record['previous_similarity']}"
    )

    print(
        f"incremental index: "
        f"{result['index_update_ms']:.3f} ms"
    )


store.save()


print("\n" + "=" * 80)
print("PERSISTENCE CHECK")
print("=" * 80)


reloaded = IncrementalVersionStore(
    path=TEST_PATH
)


print(
    "History length:",
    len(
        reloaded.history(
            "binary_search"
        )
    )
)

print(
    "Stats:",
    reloaded.stats()
)

print(
    "Stored at:",
    TEST_PATH
)

print(
    "\n✅ Incremental version indexing works."
)
