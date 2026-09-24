from src.runtime_device import get_device

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_loader import load_dev_benchmark
from src.metrics import evaluate_rankings

MODEL = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 4096

CACHE_DIR = Path("cache/qwen06b")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("Loading benchmark...")
queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]

# Reuse the exact same cached query embeddings.
query_embeddings = np.load(
    CACHE_DIR / "queries.npy"
)

print("Loading model...")
model = SentenceTransformer(
    MODEL,
    device=get_device()
)
model.max_seq_length = MAX_SEQ_LENGTH

raw_doc_texts = [
    row["text"]
    for row in corpus
]

raw_cache = CACHE_DIR / "documents_raw.npy"

if raw_cache.exists():
    print("Loading cached RAW document embeddings...")
    doc_embeddings = np.load(raw_cache)

else:
    print("Encoding 5,000 RAW documents...")
    start = time.perf_counter()

    doc_embeddings = model.encode_document(
        raw_doc_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = time.perf_counter() - start

    np.save(raw_cache, doc_embeddings)

    print(
        f"Raw document encoding completed in "
        f"{elapsed / 60:.2f} minutes."
    )

print("Computing similarity matrix...")
scores = query_embeddings @ doc_embeddings.T

TOP_K = 100
rankings = {}

for i, qid in enumerate(query_ids):
    row_scores = scores[i]

    indices = np.argpartition(
        -row_scores,
        TOP_K - 1
    )[:TOP_K]

    indices = indices[
        np.argsort(-row_scores[indices])
    ]

    rankings[qid] = [
        doc_ids[idx]
        for idx in indices
    ]

metrics = evaluate_rankings(
    rankings,
    relevant_docs
)

print("\n" + "=" * 70)
print("RAW CODE RESULTS")
print("=" * 70)

for name, value in metrics.items():
    print(f"{name:12}: {value:.6f}")

print("\nCOMPACTED BASELINE")
print("ndcg@10     : 0.821507")
print("mrr         : 0.794591")
print("recall@10   : 0.914000")
print("recall@20   : 0.940000")
print("recall@100  : 0.980000")

with open(
    "results/qwen06b_raw.json",
    "w"
) as f:
    json.dump(
        {
            "model": MODEL,
            "preprocessing": "none",
            "metrics": metrics,
        },
        f,
        indent=2
    )

print("\n✅ Raw-code ablation completed.")
