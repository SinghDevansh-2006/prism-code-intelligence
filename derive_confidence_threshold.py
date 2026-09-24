import json
from pathlib import Path

import numpy as np

from src.data_loader import (
    load_training_benchmark,
    load_dev_benchmark,
)


PERCENTILE = 30


print("=" * 80)
print("TRAIN-DERIVED CONFIDENCE THRESHOLD")
print("=" * 80)


train_queries, train_corpus, train_rel = load_training_benchmark()
val_queries, val_corpus, val_rel = load_dev_benchmark()

train_qids = [x["_id"] for x in train_queries]
val_qids = [x["_id"] for x in val_queries]

doc_ids = [x["_id"] for x in train_corpus]

doc_index = {
    did: i
    for i, did in enumerate(doc_ids)
}


# --------------------------------------------------
# LOAD EMBEDDINGS
# --------------------------------------------------

qwen_docs = np.load(
    "cache/qwen06b/documents.npy"
)

gemma_docs = np.load(
    "cache/embeddinggemma/documents.npy"
)

qwen_train = np.load(
    "cache/qwen06b/training_queries.npy"
)

gemma_train = np.load(
    "cache/embeddinggemma/training_queries.npy"
)

qwen_val = np.load(
    "cache/qwen06b/queries.npy"
)

gemma_val = np.load(
    "cache/embeddinggemma/queries.npy"
)


def row_zscore(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


def fused_scores(qwen_q, gemma_q):
    qwen = row_zscore(
        qwen_q @ qwen_docs.T
    )

    gemma = row_zscore(
        gemma_q @ gemma_docs.T
    )

    return (
        0.65 * gemma
        + 0.35 * qwen
    )


def margins(scores):
    result = []

    for row in scores:
        top2 = np.partition(
            row,
            -2
        )[-2:]

        result.append(
            float(
                np.max(top2)
                - np.min(top2)
            )
        )

    return np.array(result)


print("\nComputing training confidence margins...")

train_scores = fused_scores(
    qwen_train,
    gemma_train
)

train_margins = margins(
    train_scores
)


threshold = float(
    np.percentile(
        train_margins,
        PERCENTILE
    )
)


print("\nComputing validation margins...")

val_scores = fused_scores(
    qwen_val,
    gemma_val
)

val_margins = margins(
    val_scores
)


selected = np.where(
    val_margins <= threshold
)[0]


# --------------------------------------------------
# TRUE VALIDATION RANKS
# --------------------------------------------------

ranks = []

for i, qid in enumerate(val_qids):
    relevant_idx = doc_index[
        val_rel[qid]
    ]

    relevant_score = val_scores[
        i,
        relevant_idx
    ]

    rank = int(
        1
        + np.sum(
            val_scores[i]
            > relevant_score
        )
    )

    ranks.append(rank)


ranks = np.array(ranks)

top10_misses = ranks > 10

recoverable_11_20 = (
    (ranks > 10)
    & (ranks <= 20)
)


captured_top10_misses = np.sum(
    top10_misses[selected]
)

captured_recoverable = np.sum(
    recoverable_11_20[selected]
)


# --------------------------------------------------
# COMPARE WITH PREVIOUS VALIDATION-BATCH SELECTION
# --------------------------------------------------

previous_path = Path(
    "cache/reranker_xsmall/"
    "selective30_query_indices.npy"
)

if previous_path.exists():
    previous = np.load(
        previous_path
    )

    overlap = len(
        set(selected.tolist())
        & set(previous.tolist())
    )

else:
    previous = None
    overlap = None


print("\n" + "=" * 80)
print("FIXED ONLINE GATE")
print("=" * 80)

print(
    f"Training percentile:       {PERCENTILE}%"
)

print(
    f"Fixed margin threshold:    {threshold:.6f}"
)

print(
    f"Validation queries gated:  "
    f"{len(selected)}/1000 "
    f"({len(selected)/10:.1f}%)"
)

print(
    f"Top-10 misses captured:    "
    f"{captured_top10_misses}/"
    f"{np.sum(top10_misses)}"
)

print(
    f"Rank 11-20 captured:       "
    f"{captured_recoverable}/"
    f"{np.sum(recoverable_11_20)}"
)

if previous is not None:
    print(
        f"Overlap with old 300:      "
        f"{overlap}/{len(selected)}"
    )

    new_queries = len(
        set(selected.tolist())
        - set(previous.tolist())
    )

    print(
        f"New queries needing score: "
        f"{new_queries}"
    )


# --------------------------------------------------
# SAVE THRESHOLD
# --------------------------------------------------

result = {
    "source": "4000 training queries",
    "percentile": PERCENTILE,
    "margin_threshold": threshold,
    "validation_selected": int(
        len(selected)
    ),
    "validation_selected_fraction":
        float(len(selected) / 1000),
    "top10_misses_captured":
        int(captured_top10_misses),
    "rank11_20_captured":
        int(captured_recoverable),
}


with open(
    "results/confidence_threshold.json",
    "w"
) as f:
    json.dump(
        result,
        f,
        indent=2
    )


print(
    "\nSaved to "
    "results/confidence_threshold.json"
)

print(
    "\n✅ Train-derived online gate completed."
)
