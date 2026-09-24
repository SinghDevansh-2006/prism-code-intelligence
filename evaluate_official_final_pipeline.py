import json
import math
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import CrossEncoder

from src.code_normalizer import compact_large_literals


# ============================================================
# FROZEN SETTINGS
#
# All selected before official test scoring.
# ============================================================

GEMMA_WEIGHT = 0.65
QWEN_WEIGHT = 0.35

RERANKER_MODEL = (
    "mixedbread-ai/mxbai-rerank-xsmall-v1"
)

RERANK_TOP_K = 20
OUTPUT_TOP_K = 100
RERANK_BATCH_SIZE = 16

# Number of gated queries scored before checkpointing.
CHECKPOINT_QUERY_CHUNK = 20

CACHE_DIR = Path("cache/official")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD TRAIN-DERIVED RERANKER SETTINGS
# ============================================================

with open(
    "results/confidence_threshold.json"
) as f:
    threshold_data = json.load(f)

CONFIDENCE_THRESHOLD = float(
    threshold_data[
        "margin_threshold"
    ]
)


with open(
    "results/reranker_beta_training.json"
) as f:
    beta_data = json.load(f)

BETA = float(
    beta_data[
        "chosen"
    ][
        "beta"
    ]
)


assert abs(
    CONFIDENCE_THRESHOLD
    - 0.4298446416854859
) < 1e-12

assert abs(
    BETA - 0.20
) < 1e-12


print("=" * 80)
print("OFFICIAL FROZEN FINAL PIPELINE")
print("=" * 80)

print(
    f"\nDense weights: "
    f"{GEMMA_WEIGHT:.2f} Gemma / "
    f"{QWEN_WEIGHT:.2f} Qwen"
)

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD:.12f}"
)

print(
    f"Reranker: {RERANKER_MODEL}"
)

print(
    f"Rerank top-k: {RERANK_TOP_K}"
)

print(
    f"Reranker beta: {BETA:.2f}"
)


# ============================================================
# LOAD FROZEN EMBEDDINGS
# ============================================================

print("\nLoading frozen official embeddings...")


gemma_docs = np.load(
    CACHE_DIR
    / "embeddinggemma_documents.npy",
    mmap_mode="r",
)

gemma_queries = np.load(
    CACHE_DIR
    / "embeddinggemma_queries.npy",
    mmap_mode="r",
)

qwen_docs = np.load(
    CACHE_DIR
    / "qwen_documents.npy",
    mmap_mode="r",
)

qwen_queries = np.load(
    CACHE_DIR
    / "qwen_queries.npy",
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


assert gemma_docs.shape == (
    8765,
    768,
)

assert gemma_queries.shape == (
    3765,
    768,
)

assert qwen_docs.shape == (
    8765,
    1024,
)

assert qwen_queries.shape == (
    3765,
    1024,
)

assert len(doc_ids) == 8765
assert len(query_ids) == 3765


doc_id_to_index = {
    doc_id: idx
    for idx, doc_id
    in enumerate(doc_ids)
}


# ============================================================
# HELPERS
# ============================================================

def row_zscore(matrix):
    mean = matrix.mean(
        axis=1,
        keepdims=True,
    )

    std = (
        matrix.std(
            axis=1,
            keepdims=True,
        )
        + 1e-8
    )

    return (
        matrix - mean
    ) / std


def zscore_1d(x):
    std = x.std()

    if std < 1e-8:
        return np.zeros_like(x)

    return (
        x - x.mean()
    ) / std


def top_indices(
    scores,
    k,
):
    candidates = np.argpartition(
        -scores,
        k - 1,
        axis=1,
    )[:, :k]

    rows = np.arange(
        scores.shape[0]
    )[:, None]

    candidate_scores = scores[
        rows,
        candidates,
    ]

    order = np.argsort(
        -candidate_scores,
        axis=1,
    )

    return candidates[
        rows,
        order,
    ]


def calculate_metrics_from_orders(
    orders,
    relevant_doc_indices,
):
    total = len(orders)

    ndcg_10 = 0.0

    mrr_10 = 0.0
    mrr_100 = 0.0

    recall_10 = 0
    recall_20 = 0
    recall_100 = 0

    rank1 = 0

    ranks = []

    for i in range(total):
        order = orders[i]

        relevant_idx = (
            relevant_doc_indices[i]
        )

        positions = np.flatnonzero(
            order == relevant_idx
        )

        if len(positions) == 0:
            rank = OUTPUT_TOP_K + 1
        else:
            rank = (
                int(positions[0])
                + 1
            )

        ranks.append(rank)

        if rank == 1:
            rank1 += 1

        if rank <= 10:
            ndcg_10 += (
                1.0
                / math.log2(
                    rank + 1
                )
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

    metrics = {
        "ndcg@10":
            ndcg_10 / total,

        "mrr@10":
            mrr_10 / total,

        "mrr@100":
            mrr_100 / total,

        "recall@10":
            recall_10 / total,

        "recall@20":
            recall_20 / total,

        "recall@100":
            recall_100 / total,

        "rank1_rate":
            rank1 / total,
    }

    return metrics, ranks


# ============================================================
# BUILD FROZEN DENSE TOP-100 + CONFIDENCE MARGINS
#
# No qrels are used here.
# ============================================================

print(
    "\nBuilding dense candidates "
    "and applying fixed confidence gate..."
)


num_queries = len(query_ids)

dense_top100 = np.empty(
    (
        num_queries,
        OUTPUT_TOP_K,
    ),
    dtype=np.int32,
)

dense_top20_scores = np.empty(
    (
        num_queries,
        RERANK_TOP_K,
    ),
    dtype=np.float32,
)

margins = np.empty(
    num_queries,
    dtype=np.float32,
)


DENSE_BATCH_SIZE = 64


dense_start = time.perf_counter()


for start in range(
    0,
    num_queries,
    DENSE_BATCH_SIZE,
):
    end = min(
        start + DENSE_BATCH_SIZE,
        num_queries,
    )

    gemma_scores = (
        np.asarray(
            gemma_queries[
                start:end
            ]
        )
        @ gemma_docs.T
    )

    qwen_scores = (
        np.asarray(
            qwen_queries[
                start:end
            ]
        )
        @ qwen_docs.T
    )

    fused = (
        GEMMA_WEIGHT
        * row_zscore(
            gemma_scores
        )
        + QWEN_WEIGHT
        * row_zscore(
            qwen_scores
        )
    )


    top100 = top_indices(
        fused,
        OUTPUT_TOP_K,
    )


    dense_top100[
        start:end
    ] = top100


    rows = np.arange(
        end - start
    )[:, None]


    dense_top20_scores[
        start:end
    ] = fused[
        rows,
        top100[
            :,
            :RERANK_TOP_K
        ],
    ]


    margins[
        start:end
    ] = (
        fused[
            np.arange(
                end - start
            ),
            top100[:, 0],
        ]
        - fused[
            np.arange(
                end - start
            ),
            top100[:, 1],
        ]
    )


dense_seconds = (
    time.perf_counter()
    - dense_start
)


selected_indices = np.flatnonzero(
    margins
    <= CONFIDENCE_THRESHOLD
).astype(
    np.int32
)


selected_top20 = dense_top100[
    selected_indices,
    :RERANK_TOP_K,
].copy()


print(
    f"Queries selected by fixed gate: "
    f"{len(selected_indices)}/"
    f"{num_queries}"
)

print(
    f"Gate fraction: "
    f"{len(selected_indices) / num_queries:.3%}"
)

print(
    f"Reranker pairs required: "
    f"{len(selected_indices) * RERANK_TOP_K}"
)

print(
    f"Dense candidate preparation: "
    f"{dense_seconds:.2f} sec"
)


# ============================================================
# SAVE / VERIFY GATE DEFINITION
# ============================================================

selected_indices_path = (
    CACHE_DIR
    / "official_reranker_query_indices.npy"
)

selected_top20_path = (
    CACHE_DIR
    / "official_reranker_top20_indices.npy"
)


if selected_indices_path.exists():
    existing = np.load(
        selected_indices_path
    )

    if not np.array_equal(
        existing,
        selected_indices,
    ):
        raise RuntimeError(
            "Existing gated-query cache "
            "does not match current frozen gate."
        )

else:
    np.save(
        selected_indices_path,
        selected_indices,
    )


if selected_top20_path.exists():
    existing = np.load(
        selected_top20_path
    )

    if not np.array_equal(
        existing,
        selected_top20,
    ):
        raise RuntimeError(
            "Existing reranker candidate "
            "cache does not match current "
            "frozen dense ranking."
        )

else:
    np.save(
        selected_top20_path,
        selected_top20,
    )


# ============================================================
# LOAD RAW TEXT
#
# Still no qrels.
# ============================================================

print(
    "\nLoading source text "
    "for gated reranker candidates..."
)


corpus = load_dataset(
    "CoIR-Retrieval/apps",
    "corpus",
    split="corpus",
)

queries = load_dataset(
    "CoIR-Retrieval/apps",
    "queries",
    split="queries",
)


assert len(corpus) == 8765


raw_doc_ids = [
    row["_id"]
    for row in corpus
]


assert raw_doc_ids == doc_ids


test_query_rows = [
    row
    for row in queries
    if row[
        "partition"
    ] == "test"
]


raw_test_query_ids = [
    row["_id"]
    for row in test_query_rows
]


assert raw_test_query_ids == query_ids


query_texts = [
    row["text"]
    for row in test_query_rows
]


# Only compact documents that will actually
# be sent to the reranker.
unique_candidate_indices = sorted(
    set(
        int(idx)
        for idx
        in selected_top20.reshape(-1)
    )
)


print(
    f"Unique candidate documents "
    f"requiring reranker text: "
    f"{len(unique_candidate_indices)}"
)


compact_doc_texts = {}


for position, doc_idx in enumerate(
    unique_candidate_indices,
    start=1,
):
    compact_doc_texts[
        doc_idx
    ] = compact_large_literals(
        corpus[
            doc_idx
        ][
            "text"
        ]
    )

    if position % 1000 == 0:
        print(
            f"  Compacted "
            f"{position}/"
            f"{len(unique_candidate_indices)}"
        )


# ============================================================
# RESUMABLE RERANKER SCORE CACHE
# ============================================================

scores_path = (
    CACHE_DIR
    / "official_reranker_top20_scores.npy"
)


expected_score_shape = (
    len(selected_indices),
    RERANK_TOP_K,
)


if scores_path.exists():
    reranker_scores = np.load(
        scores_path,
        mmap_mode="r+",
    )

    if (
        reranker_scores.shape
        != expected_score_shape
    ):
        raise RuntimeError(
            "Existing reranker score cache "
            "has wrong shape: "
            f"{reranker_scores.shape}, "
            f"expected "
            f"{expected_score_shape}"
        )

else:
    reranker_scores = (
        np.lib.format.open_memmap(
            scores_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_score_shape,
        )
    )

    reranker_scores[:] = np.nan
    reranker_scores.flush()


pending_positions = [
    pos
    for pos in range(
        len(selected_indices)
    )
    if np.isnan(
        reranker_scores[
            pos
        ]
    ).any()
]


completed_before = (
    len(selected_indices)
    - len(pending_positions)
)


print(
    f"\nCached gated queries: "
    f"{completed_before}/"
    f"{len(selected_indices)}"
)

print(
    f"Pending gated queries: "
    f"{len(pending_positions)}"
)


# ============================================================
# SCORE ONLY PENDING QUERIES
# ============================================================

reranker_seconds = 0.0


if pending_positions:
    print(
        "\nLoading frozen reranker..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL,
        device="mps",
        max_length=512,
    )


    rerank_start = (
        time.perf_counter()
    )


    total_pending = len(
        pending_positions
    )


    for chunk_start in range(
        0,
        total_pending,
        CHECKPOINT_QUERY_CHUNK,
    ):
        chunk_positions = (
            pending_positions[
                chunk_start:
                chunk_start
                + CHECKPOINT_QUERY_CHUNK
            ]
        )

        pairs = []


        for score_position in (
            chunk_positions
        ):
            query_idx = int(
                selected_indices[
                    score_position
                ]
            )

            query_text = (
                query_texts[
                    query_idx
                ]
            )


            for doc_idx in (
                selected_top20[
                    score_position
                ]
            ):
                pairs.append(
                    (
                        query_text,
                        compact_doc_texts[
                            int(doc_idx)
                        ],
                    )
                )


        raw_scores = reranker.predict(
            pairs,
            batch_size=
                RERANK_BATCH_SIZE,
            show_progress_bar=True,
        )


        chunk_scores = np.asarray(
            raw_scores,
            dtype=np.float32,
        ).reshape(
            len(
                chunk_positions
            ),
            RERANK_TOP_K,
        )


        for local_pos, (
            score_position
        ) in enumerate(
            chunk_positions
        ):
            reranker_scores[
                score_position
            ] = chunk_scores[
                local_pos
            ]


        reranker_scores.flush()


        completed_now = (
            chunk_start
            + len(
                chunk_positions
            )
        )


        elapsed = (
            time.perf_counter()
            - rerank_start
        )


        rate = (
            completed_now
            / elapsed
            if elapsed > 0
            else 0.0
        )


        remaining = (
            total_pending
            - completed_now
        )


        eta_minutes = (
            remaining
            / rate
            / 60
            if rate > 0
            else 0.0
        )


        print(
            f"\nCheckpoint: "
            f"{completed_now}/"
            f"{total_pending} "
            f"pending queries scored "
            f"| ETA ~"
            f"{eta_minutes:.1f} min"
        )


    reranker_seconds = (
        time.perf_counter()
        - rerank_start
    )


else:
    print(
        "\nAll reranker scores "
        "already cached."
    )


# Ensure cache is complete.
if np.isnan(
    reranker_scores
).any():
    raise RuntimeError(
        "Reranker score cache "
        "still contains NaNs."
    )


# ============================================================
# APPLY EXACT VALIDATED BLEND
# ============================================================

print(
    "\nApplying frozen "
    "80/20 reranker blend..."
)


final_top100 = (
    dense_top100.copy()
)


query_to_score_position = {
    int(query_idx):
        score_position
    for score_position, query_idx
    in enumerate(
        selected_indices
    )
}


for query_idx in (
    selected_indices
):
    query_idx = int(
        query_idx
    )

    score_position = (
        query_to_score_position[
            query_idx
        ]
    )

    candidate_indices = (
        dense_top100[
            query_idx,
            :RERANK_TOP_K,
        ]
    )

    dense_top = (
        dense_top20_scores[
            query_idx
        ]
    )

    rerank_top = np.asarray(
        reranker_scores[
            score_position
        ]
    )


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


    rerank_positions = (
        np.argsort(
            -blended
        )
    )


    final_top100[
        query_idx,
        :RERANK_TOP_K,
    ] = candidate_indices[
        rerank_positions
    ]


# ============================================================
# ONLY NOW LOAD OFFICIAL QRELS FOR REPORTING
# ============================================================

print(
    "\nLoading official qrels "
    "for final report..."
)


qrels = load_dataset(
    "CoIR-Retrieval/apps-qrels",
    split="test",
)


relevant_doc_by_query = {
    row["query_id"]:
        row["corpus_id"]
    for row in qrels
}


assert set(
    relevant_doc_by_query
) == set(
    query_ids
)


relevant_doc_indices = np.asarray(
    [
        doc_id_to_index[
            relevant_doc_by_query[
                qid
            ]
        ]
        for qid in query_ids
    ],
    dtype=np.int32,
)


# ============================================================
# METRICS
# ============================================================

dense_metrics, dense_ranks = (
    calculate_metrics_from_orders(
        dense_top100,
        relevant_doc_indices,
    )
)


final_metrics, final_ranks = (
    calculate_metrics_from_orders(
        final_top100,
        relevant_doc_indices,
    )
)


# ============================================================
# VERIFY DENSE BASELINE AGAINST PRIOR OFFICIAL RUN
# ============================================================

with open(
    RESULTS_DIR
    / "official_dense_frozen.json"
) as f:
    previous_dense = json.load(f)


expected_dense = (
    previous_dense[
        "fusion"
    ]
)


for metric in [
    "ndcg@10",
    "mrr@10",
    "mrr@100",
    "recall@10",
    "recall@20",
    "recall@100",
    "rank1_rate",
]:
    difference = abs(
        dense_metrics[
            metric
        ]
        - expected_dense[
            metric
        ]
    )

    if difference > 1e-6:
        print(
            f"WARNING: dense metric convention differs for {metric}: "
            f"{dense_metrics[metric]:.9f} vs "
            f"{expected_dense[metric]:.9f}"
        )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 80)
print("FROZEN DENSE BASELINE")
print("=" * 80)


for name, value in (
    dense_metrics.items()
):
    print(
        f"{name:14}: "
        f"{value:.6f}"
    )


print("\n" + "=" * 80)
print("FROZEN FINAL PIPELINE")
print("=" * 80)


for name, value in (
    final_metrics.items()
):
    print(
        f"{name:14}: "
        f"{value:.6f}"
    )


print("\n" + "=" * 80)
print("DELTA")
print("=" * 80)


for metric in [
    "ndcg@10",
    "mrr@10",
    "mrr@100",
    "recall@10",
    "recall@20",
    "recall@100",
    "rank1_rate",
]:
    delta = (
        final_metrics[
            metric
        ]
        - dense_metrics[
            metric
        ]
    )

    print(
        f"{metric:14}: "
        f"{delta:+.6f}"
    )


print("\n" + "=" * 80)
print("GATING / COMPUTE")
print("=" * 80)

print(
    f"Queries reranked: "
    f"{len(selected_indices)}/"
    f"{num_queries}"
)

print(
    f"Fraction reranked: "
    f"{len(selected_indices) / num_queries:.3%}"
)

print(
    f"Queries skipping reranker: "
    f"{num_queries - len(selected_indices)}"
)

print(
    f"Reranker pairs: "
    f"{len(selected_indices) * RERANK_TOP_K}"
)

print(
    f"Fresh reranker scoring time "
    f"this run: "
    f"{reranker_seconds / 60:.2f} min"
)


# ============================================================
# SAVE FINAL RANKINGS
# ============================================================

rankings_output = {
    qid: [
        doc_ids[
            int(doc_idx)
        ]
        for doc_idx in (
            final_top100[
                query_idx
            ]
        )
    ]
    for query_idx, qid
    in enumerate(query_ids)
}


with open(
    RESULTS_DIR
    / "official_final_top100_rankings.json",
    "w",
) as f:
    json.dump(
        rankings_output,
        f,
    )


summary = {
    "evaluation": {
        "task":
            "AppsRetrieval",

        "split":
            "test",

        "queries":
            num_queries,

        "candidate_documents":
            len(doc_ids),

        "policy":
            (
                "Report-only official evaluation. "
                "All hyperparameters frozen before "
                "official qrels were scored."
            ),
    },

    "dense": {
        "embeddinggemma_weight":
            GEMMA_WEIGHT,

        "qwen_weight":
            QWEN_WEIGHT,

        "metrics":
            dense_metrics,
    },

    "confidence_gate": {
        "threshold":
            CONFIDENCE_THRESHOLD,

        "source":
            "4000 training queries",

        "queries_selected":
            int(
                len(
                    selected_indices
                )
            ),

        "fraction_selected":
            float(
                len(
                    selected_indices
                )
                / num_queries
            ),
    },

    "reranker": {
        "model":
            RERANKER_MODEL,

        "top_k":
            RERANK_TOP_K,

        "beta":
            BETA,

        "beta_source":
            (
                "training-only "
                "low-confidence sample"
            ),

        "pairs_scored":
            int(
                len(
                    selected_indices
                )
                * RERANK_TOP_K
            ),
    },

    "final_metrics":
        final_metrics,

    "delta_from_dense": {
        key:
            final_metrics[key]
            - dense_metrics[key]
        for key in final_metrics
    },
}


with open(
    RESULTS_DIR
    / "official_final_pipeline.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print("\nSaved:")

print(
    "  results/"
    "official_final_pipeline.json"
)

print(
    "  results/"
    "official_final_top100_rankings.json"
)


print(
    "\n⚠️ Official results remain report-only. "
    "Do not retune the gate, beta, fusion "
    "weight, or rerank depth from these scores."
)


print(
    "\n✅ Frozen official final-pipeline "
    "evaluation completed."
)
