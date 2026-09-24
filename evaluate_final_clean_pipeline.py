import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

from src.data_loader import load_dev_benchmark
from src.code_normalizer import compact_large_literals
from src.metrics import evaluate_rankings


MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
TOP_K = 20
BETA = 0.20
BATCH_SIZE = 16


print("=" * 80)
print("FINAL METHODOLOGY-CLEAN RETRIEVAL VALIDATION")
print("=" * 80)


# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------

queries, corpus, relevant_docs = load_dev_benchmark()

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


# ------------------------------------------------------------------
# LOAD TRAIN-DERIVED SETTINGS
# ------------------------------------------------------------------

with open(
    "results/confidence_threshold.json"
) as f:
    threshold_data = json.load(f)

threshold = threshold_data[
    "margin_threshold"
]


with open(
    "results/reranker_beta_training.json"
) as f:
    beta_data = json.load(f)

train_beta = beta_data[
    "chosen"
]["beta"]


assert abs(train_beta - BETA) < 1e-9


print(
    f"\nTrain-derived confidence threshold: "
    f"{threshold:.6f}"
)

print(
    f"Train-derived reranker beta: "
    f"{BETA:.2f}"
)


# ------------------------------------------------------------------
# LOAD DENSE EMBEDDINGS
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# APPLY FIXED ONLINE CONFIDENCE GATE
# ------------------------------------------------------------------

selected_indices = []

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
        selected_indices.append(i)


print(
    f"\nQueries selected by fixed gate: "
    f"{len(selected_indices)}/1000"
)


# ------------------------------------------------------------------
# REUSE OLD VALIDATION RERANKER CACHE
# ------------------------------------------------------------------

old_indices_path = Path(
    "cache/reranker_xsmall/"
    "selective30_query_indices.npy"
)

old_scores_path = Path(
    "cache/reranker_xsmall/"
    "selective30_top20_scores.npy"
)


old_indices = np.load(
    old_indices_path
).tolist()

old_scores = np.load(
    old_scores_path
)


score_lookup = {
    query_idx: old_scores[pos]
    for pos, query_idx
    in enumerate(old_indices)
}


missing_indices = [
    idx
    for idx in selected_indices
    if idx not in score_lookup
]


print(
    f"Cached gated queries: "
    f"{len(selected_indices) - len(missing_indices)}"
)

print(
    f"New gated queries requiring scoring: "
    f"{len(missing_indices)}"
)


# ------------------------------------------------------------------
# SCORE ONLY MISSING QUERIES
# ------------------------------------------------------------------

if missing_indices:
    pairs = []

    for query_idx in missing_indices:
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
        f"\nPrepared only {len(pairs)} "
        f"new reranker pairs."
    )

    print("Loading reranker...")

    model = CrossEncoder(
        MODEL,
        device="mps",
        max_length=512
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

    new_scores = np.asarray(
        raw_scores
    ).reshape(
        len(missing_indices),
        TOP_K
    )

    for pos, query_idx in enumerate(
        missing_indices
    ):
        score_lookup[
            query_idx
        ] = new_scores[pos]

    print(
        f"New scoring completed in "
        f"{elapsed:.2f} sec"
    )


# ------------------------------------------------------------------
# DENSE BASELINE
# ------------------------------------------------------------------

dense_rankings = {}

for i, qid in enumerate(query_ids):
    dense_rankings[qid] = [
        doc_ids[idx]
        for idx in dense_order[
            i,
            :100
        ]
    ]


dense_metrics = evaluate_rankings(
    dense_rankings,
    relevant_docs
)


# ------------------------------------------------------------------
# FINAL FIXED PIPELINE
# ------------------------------------------------------------------

def zscore_1d(x):
    std = x.std()

    if std < 1e-8:
        return np.zeros_like(x)

    return (
        x - x.mean()
    ) / std


selected_set = set(
    selected_indices
)

final_rankings = {}


for query_idx, qid in enumerate(
    query_ids
):
    original_order = dense_order[
        query_idx
    ]

    if query_idx not in selected_set:
        final_order = original_order[
            :100
        ]

    else:
        candidate_indices = (
            original_order[
                :TOP_K
            ]
        )

        dense_top = dense_scores[
            query_idx,
            candidate_indices
        ]

        rerank_top = score_lookup[
            query_idx
        ]

        dense_z = zscore_1d(
            dense_top
        )

        rerank_z = zscore_1d(
            rerank_top
        )

        blended = (
            (1.0 - BETA)
            * dense_z
            + BETA
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

    final_rankings[qid] = [
        doc_ids[idx]
        for idx in final_order
    ]


final_metrics = evaluate_rankings(
    final_rankings,
    relevant_docs
)


# ------------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("DENSE BASELINE")
print("=" * 80)

for name, value in dense_metrics.items():
    print(
        f"{name:12}: "
        f"{value:.6f}"
    )


print("\n" + "=" * 80)
print("FINAL CLEAN PIPELINE")
print("=" * 80)

for name, value in final_metrics.items():
    print(
        f"{name:12}: "
        f"{value:.6f}"
    )


print("\n" + "=" * 80)
print("IMPROVEMENT")
print("=" * 80)

print(
    f"NDCG@10: "
    f"{final_metrics['ndcg@10'] - dense_metrics['ndcg@10']:+.6f}"
)

print(
    f"MRR:     "
    f"{final_metrics['mrr'] - dense_metrics['mrr']:+.6f}"
)

print(
    f"R@10:    "
    f"{final_metrics['recall@10'] - dense_metrics['recall@10']:+.6f}"
)


result = {
    "dense_weights": {
        "embeddinggemma": 0.65,
        "qwen": 0.35
    },
    "confidence_threshold": threshold,
    "confidence_threshold_source":
        "4000 training queries",
    "reranker": MODEL,
    "reranker_beta": BETA,
    "reranker_beta_source":
        "training-only low-confidence sample",
    "rerank_top_k": TOP_K,
    "validation_queries_gated":
        len(selected_indices),
    "dense_baseline":
        dense_metrics,
    "final_clean_pipeline":
        final_metrics
}


with open(
    "results/final_clean_pipeline.json",
    "w"
) as f:
    json.dump(
        result,
        f,
        indent=2
    )


print(
    "\nSaved to "
    "results/final_clean_pipeline.json"
)

print(
    "\n✅ Final clean validation completed."
)
