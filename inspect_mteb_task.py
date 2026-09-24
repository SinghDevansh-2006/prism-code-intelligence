import mteb

task = mteb.get_task("AppsRetrieval")
task.load_data()

test_data = task.dataset["default"]["test"]

query = test_data["queries"][0]
query_id = query["id"]

relevance = test_data["relevant_docs"][query_id]
relevant_doc_id = next(iter(relevance))

relevant_doc = next(
    doc for doc in test_data["corpus"]
    if doc["id"] == relevant_doc_id
)

print("=" * 80)
print("QUERY ID:", query_id)
print("=" * 80)
print(query["text"])

print("\n" + "=" * 80)
print("CORRECT DOCUMENT ID:", relevant_doc_id)
print("=" * 80)
print(relevant_doc["text"])
