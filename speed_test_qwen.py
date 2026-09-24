import time
from sentence_transformers import SentenceTransformer
from src.data_loader import load_dev_benchmark
from src.code_normalizer import compact_large_literals

MODEL = "Qwen/Qwen3-Embedding-0.6B"

print("Loading benchmark...")
queries, corpus, _ = load_dev_benchmark()

print("Loading model...")
model = SentenceTransformer(
    MODEL,
    device="mps"
)

model.max_seq_length = 4096

sample_docs = [
    compact_large_literals(row["text"])
    for row in corpus[:250]
]

sample_queries = [
    row["text"]
    for row in queries[:100]
]

print("\nEncoding 250 documents...")
start = time.perf_counter()

doc_embeddings = model.encode_document(
    sample_docs,
    batch_size=8,
    normalize_embeddings=True,
    show_progress_bar=True
)

doc_time = time.perf_counter() - start

print("\nEncoding 100 queries...")
start = time.perf_counter()

query_embeddings = model.encode_query(
    sample_queries,
    batch_size=8,
    normalize_embeddings=True,
    show_progress_bar=True
)

query_time = time.perf_counter() - start

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print("Document embeddings:", doc_embeddings.shape)
print("Query embeddings:", query_embeddings.shape)

print(f"\n250 documents: {doc_time:.2f} seconds")
print(f"Documents/sec: {250 / doc_time:.2f}")

print(f"\n100 queries: {query_time:.2f} seconds")
print(f"Queries/sec: {100 / query_time:.2f}")

estimated_full = (5000 / (250 / doc_time)) + (1000 / (100 / query_time))

print(
    f"\nEstimated time for our full validation benchmark encoding: "
    f"{estimated_full / 60:.1f} minutes"
)

print("\n✅ Speed test completed.")
