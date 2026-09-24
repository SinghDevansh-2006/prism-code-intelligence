from src.data_loader import load_dev_benchmark
from src.code_signature import extract_code_signature

_, corpus, _ = load_dev_benchmark()

lookup = {
    row["_id"]: row["text"]
    for row in corpus
}

test_ids = [
    "d1896",
    "d3478",
    "d3047",
    "d1198",
]

for did in test_ids:
    print("\n" + "=" * 80)
    print(did)
    print("=" * 80)

    code = lookup[did]

    print("\nCODE:")
    print(code[:700])

    print("\nSIGNATURE:")
    print(
        extract_code_signature(code)
    )
