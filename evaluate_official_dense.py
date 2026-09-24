import json
import math
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset


CACHE_DIR = Path("cache/official")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TOP_K = 100
BATCH_SIZE = 64

# Frozen before official test was inspected.
GEMMA_WEIGHT = 0.65
QWEN_WEIGHT = 0.35


# ============================================================
# LOAD FROZEN EMBEDDINGS
# ============================================================

print("=" * 80)
print("OFFICIAL FROZEN DENSE EVALUATION")
print("=" * 80)

gemma_docs = np.load(
    CACHE_DIR / "embeddinggemma_documents.npy",
    mmap_mode="r",
)

gemma_queries = np.load(
    CACHE_DIR / "embeddinggemma_queries.npy",
    mmap_mode="r",
)

qwen_docs = np.load(
    CACHE_DIR / "qwen_documents.npy",
    mmap_mode="r",
)

qwen_queries = np.load(
    CACHE_DIR / "qwen_queries.npy",
    mmap_mode="r",
)


with open(
    CACHE_DIR / "document_ids.json"
) as f:
    doc_ids = json.load(f)

with open(
    CACHE_DIR / "query_ids.json"
) as f:
    query_ids = json.load(f)


assert gemma_docs.shape == (8765, 768)
assert gemma_queries.shape == (3765, 768)

assert qwen_docs.shape == (8765, 1024)
assert qwen_queries.shape == (3765, 1024)

assert len(doc_ids) == 8765
assert len(query_ids) == 3765


doc_id_to_index = {
    doc_id: i
    for i, doc_id in enumerate(doc_ids)
}


# ============================================================
# LOAD OFFICIAL QRELS
#
# This is the first scoring step using official labels.
# No model/hyperparameter choices are changed afterward.
# ============================================================

print("\nLoading official test qrels...")

qrels = load_dataset(
    "CoIR-Retrieval/apps-qrels",
    split="test",
)


relevant = {
    row["query_id"]:
        row["corpus_id"]
    for row in qrels
}


assert len(relevant) == 3765

assert set(query_ids) == set(
    relevant.keys()
)


# ============================================================
# METRICS FROM ONE RELEVANT DOCUMENT PER QUERY
# ============================================================

def calculate_metrics(ranks):
    n = len(ranks)

    ndcg_10 = 0.0

    mrr_10 = 0.0
    mrr_100 = 0.0

    recall_10 = 0
    recall_20 = 0
    recall_100 = 0

    rank_1 = 0

    for rank in ranks:
        if rank == 1:
            rank_1 += 1

        if rank <= 10:
            ndcg_10 += (
                1.0
                / math.log2(rank + 1)
            )

            mrr_10 += (
                1.0 / rank
            )

            recall_10 += 1

        if rank <= 20:
            recall_20 += 1

        if rank <= 100:
            mrr_100 += (
                1.0 / rank
            )

            recall_100 += 1

    return {
        "ndcg@10":
            ndcg_10 / n,

        "mrr@10":
            mrr_10 / n,

        "mrr@100":
            mrr_100 / n,

        "recall@10":
            recall_10 / n,

        "recall@20":
            recall_20 / n,

        "recall@100":
            recall_100 / n,

        "rank1_rate":
            rank_1 / n,
    }


# ============================================================
# HELPERS
# ============================================================

def zscore_rows(matrix):
    mean = matrix.mean(
        axis=1,
        keepdims=True,
    )

    std = matrix.std(
        axis=1,
        keepdims=True,
    ) + 1e-8

    return (
        matrix - mean
    ) / std


def top_indices(
    scores,
    k,
):
    indices = np.argpartition(
        -scores,
        k - 1,
        axis=1,
    )[:, :k]

    row_numbers = np.arange(
        scores.shape[0]
    )[:, None]

    candidate_scores = scores[
        row_numbers,
        indices,
    ]

    order = np.argsort(
        -candidate_scores,
        axis=1,
    )

    return indices[
        row_numbers,
        order,
    ]


# ============================================================
# OFFICIAL EVALUATION
# ============================================================

gemma_ranks = []
qwen_ranks = []
fusion_ranks = []

fusion_output = {}

search_start = time.perf_counter()


for start in range(
    0,
    len(query_ids),
    BATCH_SIZE,
):
    end = min(
        start + BATCH_SIZE,
        len(query_ids),
    )

    gemma_scores = (
        np.asarray(
            gemma_queries[start:end]
        )
        @ gemma_docs.T
    )

    qwen_scores = (
        np.asarray(
            qwen_queries[start:end]
        )
        @ qwen_docs.T
    )


    # Standalone ranks.
    gemma_top = top_indices(
        gemma_scores,
        TOP_K,
    )

    qwen_top = top_indices(
        qwen_scores,
        TOP_K,
    )


    # Frozen fusion.
    gemma_z = zscore_rows(
        gemma_scores
    )

    qwen_z = zscore_rows(
        qwen_scores
    )

    fused_scores = (
        GEMMA_WEIGHT
        * gemma_z
        + QWEN_WEIGHT
        * qwen_z
    )

    fusion_top = top_indices(
        fused_scores,
        TOP_K,
    )


    for local_i in range(
        end - start
    ):
        global_i = (
            start + local_i
        )

        query_id = query_ids[
            global_i
        ]

        relevant_doc = relevant[
            query_id
        ]

        relevant_idx = (
            doc_id_to_index[
                relevant_doc
            ]
        )


        # ----------------------------------------------
        # Exact rank in full 8,765-document universe.
        #
        # We compute it directly from the score of the
        # relevant document instead of truncating at 100.
        # ----------------------------------------------

        gemma_relevant_score = (
            gemma_scores[
                local_i,
                relevant_idx,
            ]
        )

        qwen_relevant_score = (
            qwen_scores[
                local_i,
                relevant_idx,
            ]
        )

        fusion_relevant_score = (
            fused_scores[
                local_i,
                relevant_idx,
            ]
        )


        gemma_rank = (
            int(
                np.count_nonzero(
                    gemma_scores[
                        local_i
                    ]
                    > gemma_relevant_score
                )
            )
            + 1
        )

        qwen_rank = (
            int(
                np.count_nonzero(
                    qwen_scores[
                        local_i
                    ]
                    > qwen_relevant_score
                )
            )
            + 1
        )

        fusion_rank = (
            int(
                np.count_nonzero(
                    fused_scores[
                        local_i
                    ]
                    > fusion_relevant_score
                )
            )
            + 1
        )


        gemma_ranks.append(
            gemma_rank
        )

        qwen_ranks.append(
            qwen_rank
        )

        fusion_ranks.append(
            fusion_rank
        )


        # ----------------------------------------------
        # SAVE FUSED TOP-100 INFERENCE RESULTS
        # ----------------------------------------------

        top_indices_row = (
            fusion_top[
                local_i
            ]
        )

        fusion_output[
            query_id
        ] = {
            doc_ids[int(idx)]:
                float(
                    fused_scores[
                        local_i,
                        idx,
                    ]
                )
            for idx in top_indices_row
        }


    print(
        f"\rProcessed "
        f"{end:4}/{len(query_ids)} "
        f"queries",
        end="",
        flush=True,
    )


search_seconds = (
    time.perf_counter()
    - search_start
)

print()


# ============================================================
# REPORT
# ============================================================

gemma_metrics = calculate_metrics(
    gemma_ranks
)

qwen_metrics = calculate_metrics(
    qwen_ranks
)

fusion_metrics = calculate_metrics(
    fusion_ranks
)


print("\n" + "=" * 80)
print("EMBEDDINGGEMMA — OFFICIAL")
print("=" * 80)

for name, value in (
    gemma_metrics.items()
):
    print(
        f"{name:14}: "
        f"{value:.6f}"
    )


print("\n" + "=" * 80)
print("QWEN — OFFICIAL")
print("=" * 80)

for name, value in (
    qwen_metrics.items()
):
    print(
        f"{name:14}: "
        f"{value:.6f}"
    )


print("\n" + "=" * 80)
print("FROZEN 65/35 FUSION — OFFICIAL")
print("=" * 80)

for name, value in (
    fusion_metrics.items()
):
    print(
        f"{name:14}: "
        f"{value:.6f}"
    )


print(
    f"\nRanking computation: "
    f"{search_seconds:.2f} seconds"
)

print(
    f"Average/query: "
    f"{search_seconds / len(query_ids) * 1000:.3f} ms"
)


# ============================================================
# SAVE
# ============================================================

summary = {
    "evaluation": {
        "task":
            "AppsRetrieval",

        "split":
            "test",

        "queries":
            3765,

        "candidate_documents":
            8765,

        "note":
            (
                "Frozen architecture evaluated "
                "after all retrieval hyperparameters "
                "were selected on development data."
            ),
    },

    "frozen_fusion": {
        "embeddinggemma_weight":
            GEMMA_WEIGHT,

        "qwen_weight":
            QWEN_WEIGHT,
    },

    "embeddinggemma":
        gemma_metrics,

    "qwen":
        qwen_metrics,

    "fusion":
        fusion_metrics,

    "ranking_seconds":
        search_seconds,
}


with open(
    RESULTS_DIR
    / "official_dense_frozen.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


with open(
    RESULTS_DIR
    / "official_dense_top100.json",
    "w",
) as f:
    json.dump(
        fusion_output,
        f,
    )


print(
    "\nSaved:"
)

print(
    "  results/official_dense_frozen.json"
)

print(
    "  results/official_dense_top100.json"
)

print(
    "\n⚠️ Official test is now report-only. "
    "Do not retune weights from these results."
)

print(
    "\n✅ Frozen official dense evaluation completed."
)
