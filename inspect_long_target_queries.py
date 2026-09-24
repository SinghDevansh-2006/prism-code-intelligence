from datasets import load_dataset

TARGET_DOCS = {"d1198", "d3047"}

qrels = load_dataset("CoIR-Retrieval/apps")["train"]
queries = load_dataset(
    "CoIR-Retrieval/apps",
    "queries",
    split="queries"
)

query_by_id = {
    row["_id"]: row
    for row in queries
}

for rel in qrels:
    if rel["corpus-id"] not in TARGET_DOCS:
        continue

    query_id = rel["query-id"]
    doc_id = rel["corpus-id"]
    query = query_by_id[query_id]

    print("\n" + "=" * 100)
    print("DOCUMENT:", doc_id)
    print("QUERY:", query_id)
    print("=" * 100)
    print(query["text"])
