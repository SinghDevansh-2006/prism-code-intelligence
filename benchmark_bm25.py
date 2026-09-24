import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.data_loader import load_dev_benchmark
from src.code_normalizer import compact_large_literals
from src.metrics import evaluate_rankings


CACHE_DIR = Path("cache/bm25")
RESULTS_DIR = Path("results")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"|\d+"
    r"|==|!=|<=|>=|//|\*\*"
    r"|[+\-*/%<>]"
)


def tokenize(text):
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
    ]


print("=" * 72)
print("BM25 LEXICAL BASELINE")
print("=" * 72)

print("\nLoading benchmark...")
queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]

print("Preparing compacted documents...")

doc_texts = [
    compact_large_literals(row["text"])
    for row in corpus
]

print("Tokenizing 5,000 documents...")

start = time.perf_counter()

tokenized_docs = [
    tokenize(text)
    for text in doc_texts
]

print(
    f"Document tokenization: "
    f"{time.perf_counter() - start:.2f} sec"
)

print("Building BM25 index...")

start = time.perf_counter()

bm25 = BM25Okapi(tokenized_docs)

print(
    f"BM25 index build: "
    f"{time.perf_counter() - start:.2f} sec"
)

print("Scoring 1,000 queries...")

start = time.perf_counter()

all_scores = np.empty(
    (len(queries), len(corpus)),
    dtype=np.float32
)

for i, query in enumerate(queries):
    tokens = tokenize(query["text"])

    all_scores[i] = bm25.get_scores(tokens)

    if (i + 1) % 100 == 0:
        print(
            f"  scored {i + 1}/{len(queries)} queries"
        )

elapsed = time.perf_counter() - start

np.save(
    CACHE_DIR / "scores.npy",
    all_scores
)

print(
    f"\nTotal BM25 query scoring time: "
    f"{elapsed:.2f} sec"
)

print(
    f"Average BM25 scoring latency/query: "
    f"{elapsed / len(queries) * 1000:.3f} ms"
)


TOP_K = 100
rankings = {}

for i, qid in enumerate(query_ids):
    row = all_scores[i]

    indices = np.argpartition(
        -row,
        TOP_K - 1
    )[:TOP_K]

    indices = indices[
        np.argsort(-row[indices])
    ]

    rankings[qid] = [
        doc_ids[idx]
        for idx in indices
    ]


metrics = evaluate_rankings(
    rankings,
    relevant_docs
)


print("\n" + "=" * 72)
print("BM25 RESULTS")
print("=" * 72)

for name, value in metrics.items():
    print(f"{name:12}: {value:.6f}")


print("\n" + "=" * 72)
print("CURRENT BEST DENSE FUSION")
print("=" * 72)

print("ndcg@10     : 0.917734")
print("mrr         : 0.900386")
print("recall@10   : 0.973000")
print("recall@100  : 0.993000")


with open(
    RESULTS_DIR / "bm25_baseline.json",
    "w"
) as f:
    json.dump(
        {
            "retriever": "BM25Okapi",
            "tokenizer": "simple_code_aware_regex",
            "preprocessing": "large_literal_compaction",
            "metrics": metrics,
            "total_query_seconds": elapsed,
        },
        f,
        indent=2,
    )

print(
    "\nSaved score matrix to "
    "cache/bm25/scores.npy"
)

print(
    "Saved metrics to "
    "results/bm25_baseline.json"
)

print("\n✅ BM25 baseline completed.")
