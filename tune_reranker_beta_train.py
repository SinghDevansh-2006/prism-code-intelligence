import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

from src.data_loader import load_training_benchmark
from src.code_normalizer import compact_large_literals
from src.metrics import evaluate_rankings


MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"

TOP_K = 20
SAMPLE_SIZE = 300
BATCH_SIZE = 16
SEED = 2026

BETAS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]

CACHE_DIR = Path("cache/reranker_xsmall")
RESULTS_DIR = Path("results")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("TRAIN-ONLY RERANKER BETA TUNING")
print("=" * 80)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

queries, corpus, relevant_docs = load_training_benchmark()

query_ids = [
    row["_id"]
    for row in queries
]

doc_ids = [
    row["_id"]
    for row in corpus
]

query_texts = {
    row["_id"]: row["text"]
    for row in queries
}

doc_texts = {
    row["_id"]: compact_large_literals(
        row["text"]
    )
    for row in corpus
}


# --------------------------------------------------
# LOAD CACHED DENSE EMBEDDINGS
# --------------------------------------------------

print("\nLoading cached embeddings...")

qwen_q = np.load(
    "cache/qwen06b/training_queries.npy"
)

qwen_d = np.load(
    "cache/qwen06b/documents.npy"
)

gemma_q = np.load(
    "cache/embeddinggemma/training_queries.npy"
)

gemma_d = np.load(
    "cache/embeddinggemma/documents.npy"
)


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

dense_scores = (
    0.65 * gemma_scores
    + 0.35 * qwen_scores
)

dense_order = np.argsort(
    -dense_scores,
    axis=1
)


# --------------------------------------------------
# LOAD TRAIN-DERIVED THRESHOLD
# --------------------------------------------------

with open(
    "results/confidence_threshold.json"
) as f:
    threshold_data = json.load(f)

threshold = threshold_data[
    "margin_threshold"
]

print(
    f"Fixed confidence threshold: "
    f"{threshold:.6f}"
)


# --------------------------------------------------
# FIND LOW-CONFIDENCE TRAINING QUERIES
# --------------------------------------------------

gated_indices = []

for i in range(len(queries)):
    row = dense_scores[i]

    top2 = np.partition(
        row,
        -2
    )[-2:]

    margin = float(
        np.max(top2)
        - np.min(top2)
    )

    if margin <= threshold:
        gated_indices.append(i)


print(
    f"Training queries passing gate: "
    f"{len(gated_indices)}/{len(queries)}"
)


# --------------------------------------------------
# FIXED RANDOM SAMPLE FROM GATED TRAINING QUERIES
# --------------------------------------------------

rng = np.random.default_rng(SEED)

sample_indices = np.sort(
    rng.choice(
        gated_indices,
        size=SAMPLE_SIZE,
        replace=False
    )
)

sample_qids = [
    query_ids[i]
    for i in sample_indices
]

sample_relevant = {
    qid: relevant_docs[qid]
    for qid in sample_qids
}


# --------------------------------------------------
# PREPARE / LOAD RERANKER SCORES
# --------------------------------------------------

score_cache = (
    CACHE_DIR
    / "training_gate_sample_scores.npy"
)

index_cache = (
    CACHE_DIR
    / "training_gate_sample_indices.npy"
)


if (
    score_cache.exists()
    and index_cache.exists()
):
    print(
        "\nLoading cached training "
        "reranker scores..."
    )

    cached_indices = np.load(
        index_cache
    )

    assert np.array_equal(
        cached_indices,
        sample_indices
    )

    reranker_scores = np.load(
        score_cache
    )

else:
    pairs = []

    for query_idx in sample_indices:
        qid = query_ids[
            query_idx
        ]

        for doc_idx in dense_order[
            query_idx,
            :TOP_K
        ]:
            did = doc_ids[
                doc_idx
            ]

            pairs.append(
                (
                    query_texts[qid],
                    doc_texts[did]
                )
            )


    print(
        f"\nPrepared {len(pairs)} "
        f"training pairs."
    )

    print("\nLoading reranker...")

    model = CrossEncoder(
        MODEL,
        device="mps",
        max_length=512
    )

    print(
        "Scoring training sample..."
    )

    start = time.perf_counter()

    raw_scores = model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    reranker_scores = np.asarray(
        raw_scores
    ).reshape(
        SAMPLE_SIZE,
        TOP_K
    )

    np.save(
        score_cache,
        reranker_scores
    )

    np.save(
        index_cache,
        sample_indices
    )

    print(
        f"\nReranker scoring time: "
        f"{elapsed / 60:.2f} min"
    )


# --------------------------------------------------
# BASELINE ON SAME TRAINING SAMPLE
# --------------------------------------------------

baseline_rankings = {}

for query_idx in sample_indices:
    qid = query_ids[
        query_idx
    ]

    baseline_rankings[qid] = [
        doc_ids[idx]
        for idx in dense_order[
            query_idx,
            :100
        ]
    ]


baseline_metrics = evaluate_rankings(
    baseline_rankings,
    sample_relevant
)


print("\n" + "=" * 80)
print("TRAIN SAMPLE DENSE BASELINE")
print("=" * 80)

for name, value in baseline_metrics.items():
    print(
        f"{name:12}: "
        f"{value:.6f}"
    )


# --------------------------------------------------
# RERANK BLEND SEARCH
# --------------------------------------------------

def zscore_1d(x):
    std = x.std()

    if std < 1e-8:
        return np.zeros_like(x)

    return (
        x - x.mean()
    ) / std


results = []

print("\n" + "=" * 80)
print("TRAIN-ONLY BETA SEARCH")
print("=" * 80)


for beta in BETAS:
    rankings = {}

    for sample_pos, query_idx in enumerate(
        sample_indices
    ):
        qid = query_ids[
            query_idx
        ]

        original_order = dense_order[
            query_idx
        ]

        candidate_indices = original_order[
            :TOP_K
        ]

        dense_top = dense_scores[
            query_idx,
            candidate_indices
        ]

        rerank_top = reranker_scores[
            sample_pos
        ]

        dense_z = zscore_1d(
            dense_top
        )

        rerank_z = zscore_1d(
            rerank_top
        )

        blended = (
            (1.0 - beta)
            * dense_z
            + beta
            * rerank_z
        )

        rerank_positions = np.argsort(
            -blended
        )

        new_top20 = candidate_indices[
            rerank_positions
        ]

        final_order = np.concatenate(
            [
                new_top20,
                original_order[
                    TOP_K:100
                ]
            ]
        )

        rankings[qid] = [
            doc_ids[idx]
            for idx in final_order
        ]


    metrics = evaluate_rankings(
        rankings,
        sample_relevant
    )

    results.append(
        {
            "beta": beta,
            **metrics
        }
    )

    print(
        f"beta={beta:.2f} | "
        f"NDCG@10="
        f"{metrics['ndcg@10']:.6f} | "
        f"MRR="
        f"{metrics['mrr']:.6f} | "
        f"R@10="
        f"{metrics['recall@10']:.6f} | "
        f"R@20="
        f"{metrics['recall@20']:.6f}"
    )


best = max(
    results,
    key=lambda x: (
        x["ndcg@10"],
        x["mrr"]
    )
)


print("\n" + "=" * 80)
print("CHOSEN TRAIN-ONLY BETA")
print("=" * 80)

print(
    f"Beta:        "
    f"{best['beta']:.2f}"
)

print(
    f"NDCG@10:    "
    f"{best['ndcg@10']:.6f}"
)

print(
    f"MRR:        "
    f"{best['mrr']:.6f}"
)

print(
    f"Recall@10:  "
    f"{best['recall@10']:.6f}"
)

print(
    f"Recall@20:  "
    f"{best['recall@20']:.6f}"
)


with open(
    RESULTS_DIR
    / "reranker_beta_training.json",
    "w"
) as f:
    json.dump(
        {
            "source": "training_only",
            "sample_size": SAMPLE_SIZE,
            "seed": SEED,
            "confidence_threshold":
                threshold,
            "baseline":
                baseline_metrics,
            "results":
                results,
            "chosen":
                best
        },
        f,
        indent=2
    )


print(
    "\nSaved to "
    "results/reranker_beta_training.json"
)

print(
    "\n✅ Training-only beta tuning completed."
)
