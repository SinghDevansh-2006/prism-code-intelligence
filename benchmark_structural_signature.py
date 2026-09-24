import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.data_loader import load_dev_benchmark
from src.code_signature import extract_code_signature
from src.metrics import evaluate_rankings


MODEL = "google/embeddinggemma-300m"
BATCH_SIZE = 16

CACHE_DIR = Path("cache/structural_signature")
RESULTS_DIR = Path("results")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 78)
print("STRUCTURAL SIGNATURE RETRIEVAL")
print("=" * 78)


# --------------------------------------------------
# LOAD BENCHMARK
# --------------------------------------------------

print("\nLoading benchmark...")

queries, corpus, relevant_docs = load_dev_benchmark()

query_ids = [row["_id"] for row in queries]
doc_ids = [row["_id"] for row in corpus]


# --------------------------------------------------
# EXTRACT STRUCTURAL SIGNATURES
# --------------------------------------------------

print("Extracting AST signatures...")

start = time.perf_counter()

signatures = [
    extract_code_signature(row["text"])
    for row in corpus
]

elapsed = time.perf_counter() - start

unparsed = sum(
    sig == "unparsed python code"
    for sig in signatures
)

lengths = np.array([
    len(sig)
    for sig in signatures
])

print(
    f"Signature extraction: {elapsed:.2f} sec"
)

print(
    f"Unparsed documents:   {unparsed}/{len(corpus)}"
)

print(
    f"Median signature chars: {np.median(lengths):.0f}"
)

print(
    f"Max signature chars:    {np.max(lengths)}"
)


# --------------------------------------------------
# LOAD EXISTING EMBEDDINGS
# --------------------------------------------------

print("\nLoading cached base embeddings...")

qwen_q = np.load(
    "cache/qwen06b/queries.npy"
)

qwen_d = np.load(
    "cache/qwen06b/documents.npy"
)

gemma_q = np.load(
    "cache/embeddinggemma/queries.npy"
)

gemma_d = np.load(
    "cache/embeddinggemma/documents.npy"
)


# --------------------------------------------------
# ENCODE SIGNATURES
# --------------------------------------------------

sig_cache = CACHE_DIR / "documents.npy"

if sig_cache.exists():
    print(
        "Loading cached signature embeddings..."
    )

    signature_embeddings = np.load(
        sig_cache
    )

else:
    print("\nLoading EmbeddingGemma...")

    model = SentenceTransformer(
        MODEL,
        device="mps",
        model_kwargs={
            "torch_dtype": torch.float32
        }
    )

    model.max_seq_length = 2048

    print(
        "Encoding 5,000 structural signatures..."
    )

    start = time.perf_counter()

    signature_embeddings = model.encode_document(
        signatures,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = time.perf_counter() - start

    np.save(
        sig_cache,
        signature_embeddings
    )

    print(
        f"Signature encoding time: "
        f"{elapsed / 60:.2f} min"
    )


print(
    "\nSignature matrix:",
    signature_embeddings.shape
)


# --------------------------------------------------
# SCORE MATRICES
# --------------------------------------------------

def row_zscore(x):
    mean = x.mean(
        axis=1,
        keepdims=True
    )

    std = (
        x.std(
            axis=1,
            keepdims=True
        )
        + 1e-8
    )

    return (
        x - mean
    ) / std


qwen_scores = row_zscore(
    qwen_q @ qwen_d.T
)

gemma_scores = row_zscore(
    gemma_q @ gemma_d.T
)

base_dense = (
    0.65 * gemma_scores
    + 0.35 * qwen_scores
)

base_dense = row_zscore(
    base_dense
)

signature_scores = row_zscore(
    gemma_q @ signature_embeddings.T
)


# --------------------------------------------------
# HELPER
# --------------------------------------------------

def evaluate_scores(scores):
    rankings = {}

    for i, qid in enumerate(query_ids):
        row = scores[i]

        indices = np.argpartition(
            -row,
            99
        )[:100]

        indices = indices[
            np.argsort(
                -row[indices]
            )
        ]

        rankings[qid] = [
            doc_ids[idx]
            for idx in indices
        ]

    return evaluate_rankings(
        rankings,
        relevant_docs
    )


# --------------------------------------------------
# SIGNATURE-ONLY RESULT
# --------------------------------------------------

signature_metrics = evaluate_scores(
    signature_scores
)

print("\n" + "=" * 78)
print("STRUCTURAL SIGNATURE ONLY")
print("=" * 78)

for name, value in signature_metrics.items():
    print(
        f"{name:12}: {value:.6f}"
    )


# --------------------------------------------------
# FUSION
# --------------------------------------------------

weights = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]

results = []

print("\n" + "=" * 78)
print("DENSE + STRUCTURAL FUSION")
print("gamma = structural-signature weight")
print("=" * 78)

for gamma in weights:
    fused = (
        (1.0 - gamma) * base_dense
        + gamma * signature_scores
    )

    metrics = evaluate_scores(
        fused
    )

    results.append({
        "gamma": gamma,
        **metrics
    })

    print(
        f"gamma={gamma:.2f} | "
        f"NDCG@10={metrics['ndcg@10']:.6f} | "
        f"MRR={metrics['mrr']:.6f} | "
        f"R@10={metrics['recall@10']:.6f} | "
        f"R@20={metrics['recall@20']:.6f} | "
        f"R@100={metrics['recall@100']:.6f}"
    )


best = max(
    results,
    key=lambda x: (
        x["ndcg@10"],
        x["mrr"]
    )
)


print("\n" + "=" * 78)
print("BEST STRUCTURAL FUSION")
print("=" * 78)

print(
    f"Structural weight: {best['gamma']:.2f}"
)

print(
    f"NDCG@10:          {best['ndcg@10']:.6f}"
)

print(
    f"MRR:              {best['mrr']:.6f}"
)

print(
    f"Recall@10:        {best['recall@10']:.6f}"
)

print(
    f"Recall@20:        {best['recall@20']:.6f}"
)

print(
    f"Recall@100:       {best['recall@100']:.6f}"
)


print("\nCurrent dense reference:")
print("NDCG@10:          0.917734")
print("MRR:              0.900386")
print("Recall@10:        0.973000")
print("Recall@20:        0.984000")
print("Recall@100:       0.993000")


with open(
    RESULTS_DIR / "structural_signature.json",
    "w"
) as f:
    json.dump(
        {
            "signature_model": MODEL,
            "unparsed_documents": unparsed,
            "signature_only": signature_metrics,
            "fusion_results": results,
            "best": best
        },
        f,
        indent=2
    )


print(
    "\nSaved results to "
    "results/structural_signature.json"
)

print(
    "\n✅ Structural-signature benchmark completed."
)
