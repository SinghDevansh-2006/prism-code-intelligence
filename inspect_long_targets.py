from datasets import load_dataset

TARGETS = {"d3047", "d1198"}

corpus = load_dataset(
    "CoIR-Retrieval/apps",
    "corpus",
    split="corpus"
)

for row in corpus:
    if row["_id"] not in TARGETS:
        continue

    text = row["text"]
    lines = text.splitlines()

    print("\n" + "=" * 100)
    print("DOCUMENT:", row["_id"])
    print("Characters:", len(text))
    print("Lines:", len(lines))
    print("=" * 100)

    print("\nFIRST 25 LINES:")
    for i, line in enumerate(lines[:25], 1):
        print(f"{i:4}: {line}")

    print("\nLAST 15 LINES:")
    start = max(1, len(lines) - 14)
    for i, line in enumerate(lines[-15:], start):
        print(f"{i:4}: {line}")
