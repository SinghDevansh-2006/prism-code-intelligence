from src.runtime_device import get_device

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
SELECT_FRACTION = 0.30
BATCH_SIZE = 16

BETAS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
]


CACHE_DIR = Path("cache/reranker_xsmall")
RESULTS_DIR = Path("results")

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 78)
print("SELECTIVE RERANKER — FULL VALIDATION")
print("=" * 78)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading benchmark...")

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


# --------------------------------------------------
# LOAD DENSE EMBEDDINGS
# --------------------------------------------------

print("Loading cached embeddings...")

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


# --------------------------------------------------
# FULL DENSE ORDER
# --------------------------------------------------

print("Building dense rankings...")

dense_order = np.argsort(
    -dense_scores,
    axis=1
)


# --------------------------------------------------
# CONFIDENCE MARGIN
# --------------------------------------------------

margins = []

for i in range(len(queries)):
    row = dense_scores[i]

    top2 = np.partition(
        row,
        -2
    )[-2:]

    top2.sort()

    margin = float(
        top2[-1] - top2[-2]
    )

    margins.append(
        (margin, i)
    )


margins.sort(
    key=lambda x: x[0]
)

select_count = int(
    len(queries)
    * SELECT_FRACTION
)

selected = margins[
    :select_count
]

selected_indices = [
    i
    for _, i in selected
]


print(
    f"\nSelected {select_count}/"
    f"{len(queries)} "
    f"lowest-confidence queries "
    f"({SELECT_FRACTION:.0%})."
)


# --------------------------------------------------
# BASELINE METRICS
# --------------------------------------------------

dense_rankings = {}

for i, qid in enumerate(query_ids):
    dense_rankings[qid] = [
        doc_ids[idx]
        for idx in dense_order[i, :100]
    ]


dense_metrics = evaluate_rankings(
    dense_rankings,
    relevant_docs
)


print("\n" + "=" * 78)
print("DENSE BASELINE")
print("=" * 78)

for name, value in dense_metrics.items():
    print(
        f"{name:12}: "
        f"{value:.6f}"
    )


# --------------------------------------------------
# PREPARE SELECTIVE PAIRS
# --------------------------------------------------

score_cache = (
    CACHE_DIR
    / "selective30_top20_scores.npy"
)

index_cache = (
    CACHE_DIR
    / "selective30_query_indices.npy"
)


if (
    score_cache.exists()
    and index_cache.exists()
):
    print(
        "\nLoading cached selective "
        "reranker scores..."
    )

    reranker_scores = np.load(
        score_cache
    )

    selected_indices = np.load(
        index_cache
    ).tolist()

else:
    pairs = []

    for query_idx in selected_indices:
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
                    doc_texts[did],
                )
            )


    print(
        f"\nPrepared {len(pairs)} "
        f"reranker pairs."
    )

    print("\nLoading reranker...")

    model = CrossEncoder(
        MODEL,
        device=get_device(),
        max_length=512,
    )

    print("Scoring selected queries...")

    start = time.perf_counter()

    raw_scores = model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    reranker_scores = np.asarray(
        raw_scores
    ).reshape(
        select_count,
        TOP_K
    )

    np.save(
        score_cache,
        reranker_scores,
    )

    np.save(
        index_cache,
        np.array(
            selected_indices,
            dtype=np.int32
        ),
    )

    print(
        f"\nSelective reranking time: "
        f"{elapsed / 60:.2f} min"
    )

    print(
        f"Average selected-query time: "
        f"{elapsed / select_count:.3f} sec"
    )


# --------------------------------------------------
# BLENDING
# --------------------------------------------------

def zscore_1d(x):
    std = x.std()

    if std < 1e-8:
        return np.zeros_like(
            x
        )

    return (
        x - x.mean()
    ) / std


selected_position = {
    query_idx: pos
    for pos, query_idx
    in enumerate(
        selected_indices
    )
}


results = []

print("\n" + "=" * 78)
print("SELECTIVE RERANK RESULTS")
print(
    "Only lowest-confidence 30% "
    "are reranked"
)
print("=" * 78)


for beta in BETAS:

    rankings = {}

    for query_idx, qid in enumerate(
        query_ids
    ):
        original_order = dense_order[
            query_idx
        ]

        if query_idx not in selected_position:
            final_order = original_order[
                :100
            ]

        else:
            sample_pos = (
                selected_position[
                    query_idx
                ]
            )

            candidate_indices = (
                original_order[
                    :TOP_K
                ]
            )

            dense_top = (
                dense_scores[
                    query_idx,
                    candidate_indices
                ]
            )

            rerank_top = (
                reranker_scores[
                    sample_pos
                ]
            )

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

            rerank_positions = (
                np.argsort(
                    -blended
                )
            )

            new_top20 = (
                candidate_indices[
                    rerank_positions
                ]
            )

            final_order = (
                np.concatenate(
                    [
                        new_top20,
                        original_order[
                            TOP_K:100
                        ],
                    ]
                )
            )

        rankings[qid] = [
            doc_ids[idx]
            for idx in final_order
        ]


    metrics = evaluate_rankings(
        rankings,
        relevant_docs
    )

    results.append(
        {
            "beta": beta,
            **metrics,
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
        f"{metrics['recall@20']:.6f} | "
        f"R@100="
        f"{metrics['recall@100']:.6f}"
    )


# --------------------------------------------------
# BEST
# --------------------------------------------------

best = max(
    results,
    key=lambda x: (
        x["ndcg@10"],
        x["mrr"],
    )
)


print("\n" + "=" * 78)
print("BEST SELECTIVE RESULT")
print("=" * 78)

print(
    f"Reranker beta: "
    f"{best['beta']:.2f}"
)

print(
    f"NDCG@10:       "
    f"{best['ndcg@10']:.6f}"
)

print(
    f"MRR:           "
    f"{best['mrr']:.6f}"
)

print(
    f"Recall@10:     "
    f"{best['recall@10']:.6f}"
)

print(
    f"Recall@20:     "
    f"{best['recall@20']:.6f}"
)

print(
    f"Recall@100:    "
    f"{best['recall@100']:.6f}"
)


print("\nDense reference:")

print(
    f"NDCG@10:       "
    f"{dense_metrics['ndcg@10']:.6f}"
)

print(
    f"MRR:           "
    f"{dense_metrics['mrr']:.6f}"
)

print(
    f"Recall@10:     "
    f"{dense_metrics['recall@10']:.6f}"
)


with open(
    RESULTS_DIR
    / "selective_reranker.json",
    "w",
) as f:
    json.dump(
        {
            "model": MODEL,
            "selection_fraction":
                SELECT_FRACTION,
            "top_k": TOP_K,
            "dense_baseline":
                dense_metrics,
            "results":
                results,
            "best":
                best,
        },
        f,
        indent=2,
    )


print(
    "\nSaved to "
    "results/selective_reranker.json"
)

print(
    "\n✅ Selective reranker "
    "evaluation completed."
)
