import json
import time
from pathlib import Path

from src.structural_metadata import (
    extract_structural_metadata,
)


RUNTIME_DIR = Path(
    "runtime_index"
)

CORPUS_PATH = (
    RUNTIME_DIR
    / "corpus.jsonl"
)

OUTPUT_PATH = (
    RUNTIME_DIR
    / "structural_index.jsonl"
)


print("=" * 80)
print("BUILDING STRUCTURAL CODE INDEX")
print("=" * 80)


documents = []

with open(CORPUS_PATH) as f:
    for line in f:
        documents.append(
            json.loads(line)
        )


print(
    f"\nDocuments: {len(documents)}"
)


start = time.perf_counter()

parsed_count = 0
unparsed_count = 0

recursive_docs = 0
nested_loop_docs = 0

records = []


for i, document in enumerate(
    documents,
    start=1
):
    metadata = (
        extract_structural_metadata(
            document["code"]
        )
    )

    if metadata["parsed"]:
        parsed_count += 1
    else:
        unparsed_count += 1

    if metadata[
        "recursive_functions"
    ]:
        recursive_docs += 1

    if metadata[
        "max_loop_depth"
    ] >= 2:
        nested_loop_docs += 1

    record = {
        "id": document["id"],
        **metadata,
    }

    records.append(record)

    if i % 1000 == 0:
        print(
            f"  indexed {i}/"
            f"{len(documents)}"
        )


with open(
    OUTPUT_PATH,
    "w"
) as f:

    for record in records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


elapsed = (
    time.perf_counter()
    - start
)


print("\n" + "=" * 80)
print("STRUCTURAL INDEX READY")
print("=" * 80)

print(
    f"Parsed successfully: "
    f"{parsed_count}"
)

print(
    f"Fallback/unparsed:   "
    f"{unparsed_count}"
)

print(
    f"Recursive docs:      "
    f"{recursive_docs}"
)

print(
    f"Nested-loop docs:    "
    f"{nested_loop_docs}"
)

print(
    f"Build time:          "
    f"{elapsed:.2f} sec"
)

print(
    f"Index path:          "
    f"{OUTPUT_PATH}"
)

print(
    "\n✅ Structural index built."
)
