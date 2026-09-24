from datasets import load_dataset

qrels = load_dataset("CoIR-Retrieval/apps")
queries = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")

first = qrels["train"][0]

query_id = first["query-id"]
doc_id = first["corpus-id"]

query = next(x for x in queries if x["_id"] == query_id)
doc = next(x for x in corpus if x["_id"] == doc_id)

print("QUERY ID:", query_id)
print("QUERY PARTITION:", query["partition"])

print("\nDOCUMENT ID:", doc_id)
print("DOCUMENT PARTITION:", doc["partition"])

print("\nQUERY PREVIEW:")
print(query["text"][:1200])

print("\n" + "=" * 80)

print("CORRECT CODE:")
print(doc["text"])
