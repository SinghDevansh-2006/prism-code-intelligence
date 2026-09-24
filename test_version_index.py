from src.version_index import VersionIndex


index = VersionIndex(
    similarity_threshold=0.75
)


index.add_snapshot(
    logical_id="binary_search",
    version="v1",
    code="""
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
)


index.add_snapshot(
    logical_id="binary_search",
    version="v2",
    code="""
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
)


index.add_snapshot(
    logical_id="binary_search",
    version="v3",
    code="""
from bisect import bisect_left

def binary_search(arr, target):
    pos = bisect_left(arr, target)

    if pos < len(arr) and arr[pos] == target:
        return pos

    return -1
""",
)


index.add_snapshot(
    logical_id="factorial",
    version="v1",
    code="""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
)


index.add_snapshot(
    logical_id="factorial",
    version="v2",
    code="""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
)


index.build_lineages()


print("=" * 80)
print("VERSION-AWARE INDEX TEST")
print("=" * 80)

print("\nStats:")
for key, value in index.stats().items():
    print(
        f"{key:28}: {value}"
    )


print("\nBinary-search history:")

for item in index.history(
    "binary_search"
):
    print(
        f"{item['version']:3} | "
        f"{item['change_type']:22} | "
        f"previous similarity="
        f"{item['previous_similarity']}"
    )


print("\nFactorial history:")

for item in index.history(
    "factorial"
):
    print(
        f"{item['version']:3} | "
        f"{item['change_type']:22} | "
        f"previous similarity="
        f"{item['previous_similarity']}"
    )


path = index.save(
    "runtime_index/version_index_demo.json"
)

print(
    "\nSaved:",
    path
)

print(
    "\n✅ Version-aware indexing foundation works."
)
