from datasets import load_dataset

qrels = load_dataset("CoIR-Retrieval/apps")["train"]

queries = load_dataset(
    "CoIR-Retrieval/apps",
    "queries",
    split="queries"
)

corpus = load_dataset(
    "CoIR-Retrieval/apps",
    "corpus",
    split="corpus"
)

query_by_id = {
    row["_id"]: row
    for row in queries
    if row["partition"] == "train"
}

doc_by_id = {
    row["_id"]: row
    for row in corpus
    if row["partition"] == "train"
}


def clean_url(row):
    meta = row.get("meta_information") or {}
    url = meta.get("url") or ""
    return url.strip()


matched = []
mismatched = []
missing = []

for rel in qrels:
    qid = rel["query-id"]
    did = rel["corpus-id"]

    q = query_by_id[qid]
    d = doc_by_id[did]

    q_url = clean_url(q)
    d_url = clean_url(d)

    if not q_url or not d_url:
        missing.append((qid, did, q_url, d_url))

    elif q_url == d_url:
        matched.append((qid, did, q_url))

    else:
        mismatched.append((qid, did, q_url, d_url))


total = len(qrels)

print("=" * 80)
print("TRAIN LABEL ALIGNMENT")
print("=" * 80)

print("Total pairs:", total)
print("Matching URLs:", len(matched))
print("Mismatching URLs:", len(mismatched))
print("Missing URL(s):", len(missing))

print(
    f"\nExact URL agreement: "
    f"{len(matched) / total * 100:.2f}%"
)

print("\nFIRST 20 MISMATCHES")
print("=" * 80)

for qid, did, q_url, d_url in mismatched[:20]:
    print(f"\n{qid} -> {did}")
    print("Query URL:", q_url)
    print("Doc URL:  ", d_url)

print("\nCHECK OUR FIVE WORST CASES")
print("=" * 80)

targets = {"q1874", "q1156", "q1973", "q3174", "q1840"}

for rel in qrels:
    if rel["query-id"] not in targets:
        continue

    qid = rel["query-id"]
    did = rel["corpus-id"]

    q = query_by_id[qid]
    d = doc_by_id[did]

    print(f"\n{qid} -> {did}")
    print("Query URL:", clean_url(q))
    print("Doc URL:  ", clean_url(d))
    print(
        "MATCH:",
        clean_url(q) == clean_url(d)
    )
