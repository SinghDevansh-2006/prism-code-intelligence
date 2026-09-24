import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import (
    load_training_benchmark,
    load_dev_benchmark,
)


TOP_K = 100
SEED = 123

BETAS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
]

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("LEARNED ADAPTIVE DENSE FUSION")
print("=" * 80)


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------

print("\nLoading train + validation data...")

train_queries, train_corpus, train_rel = load_training_benchmark()
val_queries, val_corpus, val_rel = load_dev_benchmark()

train_qids = [x["_id"] for x in train_queries]
val_qids = [x["_id"] for x in val_queries]

doc_ids = [x["_id"] for x in train_corpus]
val_doc_ids = [x["_id"] for x in val_corpus]

assert doc_ids == val_doc_ids

doc_index = {
    did: i
    for i, did in enumerate(doc_ids)
}


# ------------------------------------------------------------------
# VERIFY TRAINING QUERY CACHE ORDER
# ------------------------------------------------------------------

with open("cache/training_query_ids.json") as f:
    cached_train_qids = json.load(f)

assert cached_train_qids == train_qids, (
    "Training query embedding order does not match loader order."
)


# ------------------------------------------------------------------
# LOAD EMBEDDINGS
# ------------------------------------------------------------------

print("Loading cached embeddings...")

qwen_docs = np.load(
    "cache/qwen06b/documents.npy"
)

gemma_docs = np.load(
    "cache/embeddinggemma/documents.npy"
)

qwen_train_q = np.load(
    "cache/qwen06b/training_queries.npy"
)

gemma_train_q = np.load(
    "cache/embeddinggemma/training_queries.npy"
)

qwen_val_q = np.load(
    "cache/qwen06b/queries.npy"
)

gemma_val_q = np.load(
    "cache/embeddinggemma/queries.npy"
)


# ------------------------------------------------------------------
# SCORE MATRICES
# ------------------------------------------------------------------

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


print("Computing training score matrices...")

train_qwen = row_zscore(
    qwen_train_q @ qwen_docs.T
)

train_gemma = row_zscore(
    gemma_train_q @ gemma_docs.T
)

print("Computing validation score matrices...")

val_qwen = row_zscore(
    qwen_val_q @ qwen_docs.T
)

val_gemma = row_zscore(
    gemma_val_q @ gemma_docs.T
)


# ------------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------------

FEATURE_NAMES = [
    "gemma_z",
    "qwen_z",
    "dense_fusion",
    "gemma_minus_qwen",
    "abs_gemma_minus_qwen",
    "gemma_times_qwen",
    "dense_rank_feature",
    "query_length_log",
    "gemma_top_margin",
    "qwen_top_margin",
    "dense_top_margin",
]


def top_margin(row):
    values = np.partition(
        row,
        -2
    )[-2:]

    return float(
        np.max(values)
        - np.min(values)
    )


def zscore_1d(x):
    std = x.std()

    if std < 1e-8:
        return np.zeros_like(x)

    return (
        x - x.mean()
    ) / std


def build_groups(
    queries,
    relevant_docs,
    gemma_scores,
    qwen_scores,
):
    groups = []

    missing_positive = 0

    for i, query in enumerate(queries):
        qid = query["_id"]

        gemma_row = gemma_scores[i]
        qwen_row = qwen_scores[i]

        dense_row = (
            0.65 * gemma_row
            + 0.35 * qwen_row
        )

        candidate_indices = np.argpartition(
            -dense_row,
            TOP_K - 1
        )[:TOP_K]

        candidate_indices = candidate_indices[
            np.argsort(
                -dense_row[candidate_indices]
            )
        ]

        gemma_candidate = gemma_row[
            candidate_indices
        ]

        qwen_candidate = qwen_row[
            candidate_indices
        ]

        dense_candidate = dense_row[
            candidate_indices
        ]

        rank_feature = (
            1.0
            / np.log2(
                np.arange(
                    2,
                    TOP_K + 2
                )
            )
        )

        query_length_log = np.log1p(
            len(
                query["text"].split()
            )
        )

        gemma_margin = top_margin(
            gemma_row
        )

        qwen_margin = top_margin(
            qwen_row
        )

        dense_margin = top_margin(
            dense_row
        )

        features = np.column_stack(
            [
                gemma_candidate,
                qwen_candidate,
                dense_candidate,
                gemma_candidate
                - qwen_candidate,
                np.abs(
                    gemma_candidate
                    - qwen_candidate
                ),
                gemma_candidate
                * qwen_candidate,
                rank_feature,
                np.full(
                    TOP_K,
                    query_length_log
                ),
                np.full(
                    TOP_K,
                    gemma_margin
                ),
                np.full(
                    TOP_K,
                    qwen_margin
                ),
                np.full(
                    TOP_K,
                    dense_margin
                ),
            ]
        ).astype(np.float32)

        positive_doc = relevant_docs[qid]
        positive_idx = doc_index[
            positive_doc
        ]

        positions = np.where(
            candidate_indices
            == positive_idx
        )[0]

        if len(positions) == 0:
            target_position = -1
            missing_positive += 1

        else:
            target_position = int(
                positions[0]
            )

        groups.append(
            {
                "qid": qid,
                "candidate_indices":
                    candidate_indices,
                "features": features,
                "dense_scores":
                    dense_candidate.astype(
                        np.float32
                    ),
                "target_position":
                    target_position,
            }
        )

    return groups, missing_positive


print("\nBuilding training candidate groups...")

train_groups, train_missing = build_groups(
    train_queries,
    train_rel,
    train_gemma,
    train_qwen,
)

print("Building validation candidate groups...")

val_groups, val_missing = build_groups(
    val_queries,
    val_rel,
    val_gemma,
    val_qwen,
)


print("\nCandidate coverage:")
print(
    f"Training relevant doc in top {TOP_K}: "
    f"{len(train_groups) - train_missing}/"
    f"{len(train_groups)}"
)

print(
    f"Validation relevant doc in top {TOP_K}: "
    f"{len(val_groups) - val_missing}/"
    f"{len(val_groups)}"
)


# ------------------------------------------------------------------
# QUERY-LEVEL INNER SPLIT
# ------------------------------------------------------------------

recoverable_train = [
    g
    for g in train_groups
    if g["target_position"] >= 0
]

rng = np.random.default_rng(SEED)

indices = np.arange(
    len(recoverable_train)
)

rng.shuffle(indices)

split_at = int(
    0.80
    * len(indices)
)

inner_train_idx = indices[
    :split_at
]

inner_holdout_idx = indices[
    split_at:
]

inner_train = [
    recoverable_train[i]
    for i in inner_train_idx
]

inner_holdout = [
    recoverable_train[i]
    for i in inner_holdout_idx
]


print("\nInner learning split:")
print(
    "Inner train queries:",
    len(inner_train)
)

print(
    "Inner holdout queries:",
    len(inner_holdout)
)


# ------------------------------------------------------------------
# FLATTEN GROUPS
# ------------------------------------------------------------------

def flatten_groups(groups):
    X_parts = []
    y_parts = []

    for group in groups:
        X = group["features"]

        y = np.zeros(
            len(X),
            dtype=np.int8
        )

        y[
            group["target_position"]
        ] = 1

        X_parts.append(X)
        y_parts.append(y)

    return (
        np.vstack(X_parts),
        np.concatenate(y_parts),
    )


X_inner, y_inner = flatten_groups(
    inner_train
)

weights_inner = compute_sample_weight(
    class_weight="balanced",
    y=y_inner
)


# ------------------------------------------------------------------
# MODEL FACTORY
# ------------------------------------------------------------------

def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=SEED,
    )


# ------------------------------------------------------------------
# TRAIN INNER MODEL
# ------------------------------------------------------------------

print("\nTraining inner adaptive fusion model...")

inner_model = make_model()

inner_model.fit(
    X_inner,
    y_inner,
    sample_weight=weights_inner,
)


# ------------------------------------------------------------------
# METRIC HELPER
# ------------------------------------------------------------------

def evaluate_group_scores(
    groups,
    model,
    beta,
):
    ndcg_values = []
    rr_values = []
    recall10_values = []

    for group in groups:
        target = group[
            "target_position"
        ]

        if target < 0:
            ndcg_values.append(0.0)
            rr_values.append(0.0)
            recall10_values.append(0.0)
            continue

        learned = model.predict_proba(
            group["features"]
        )[:, 1]

        learned = zscore_1d(
            learned
        )

        dense = zscore_1d(
            group["dense_scores"]
        )

        combined = (
            (1.0 - beta) * dense
            + beta * learned
        )

        order = np.argsort(
            -combined
        )

        rank = int(
            np.where(
                order == target
            )[0][0]
            + 1
        )

        if rank <= 10:
            ndcg_values.append(
                1.0
                / np.log2(
                    rank + 1
                )
            )

            recall10_values.append(
                1.0
            )

        else:
            ndcg_values.append(
                0.0
            )

            recall10_values.append(
                0.0
            )

        rr_values.append(
            1.0 / rank
        )

    return {
        "ndcg@10":
            float(
                np.mean(
                    ndcg_values
                )
            ),
        "mrr":
            float(
                np.mean(
                    rr_values
                )
            ),
        "recall@10":
            float(
                np.mean(
                    recall10_values
                )
            ),
    }


# ------------------------------------------------------------------
# INNER BETA SELECTION
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("INNER HOLDOUT — BLEND SEARCH")
print("=" * 80)

inner_results = []

for beta in BETAS:
    metrics = evaluate_group_scores(
        inner_holdout,
        inner_model,
        beta,
    )

    inner_results.append(
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
        f"{metrics['recall@10']:.6f}"
    )


best_inner = max(
    inner_results,
    key=lambda x: (
        x["ndcg@10"],
        x["mrr"],
    )
)

chosen_beta = best_inner[
    "beta"
]


print("\nChosen beta from inner holdout:")
print(
    f"beta = {chosen_beta:.2f}"
)


# ------------------------------------------------------------------
# RETRAIN ON ALL 4K TRAINING QUERIES THAT ARE RECOVERABLE
# ------------------------------------------------------------------

print("\nTraining final model on all recoverable training queries...")

X_full, y_full = flatten_groups(
    recoverable_train
)

weights_full = compute_sample_weight(
    class_weight="balanced",
    y=y_full
)

final_model = make_model()

final_model.fit(
    X_full,
    y_full,
    sample_weight=weights_full,
)


# ------------------------------------------------------------------
# OUTER VALIDATION — ONCE
# ------------------------------------------------------------------

baseline_metrics = evaluate_group_scores(
    val_groups,
    final_model,
    0.0,
)

learned_metrics = evaluate_group_scores(
    val_groups,
    final_model,
    chosen_beta,
)


print("\n" + "=" * 80)
print("OUTER 1,000-QUERY VALIDATION")
print("=" * 80)

print("\nDense baseline:")
print(
    f"NDCG@10:   "
    f"{baseline_metrics['ndcg@10']:.6f}"
)

print(
    f"MRR:       "
    f"{baseline_metrics['mrr']:.6f}"
)

print(
    f"Recall@10: "
    f"{baseline_metrics['recall@10']:.6f}"
)


print("\nLearned adaptive fusion:")
print(
    f"NDCG@10:   "
    f"{learned_metrics['ndcg@10']:.6f}"
)

print(
    f"MRR:       "
    f"{learned_metrics['mrr']:.6f}"
)

print(
    f"Recall@10: "
    f"{learned_metrics['recall@10']:.6f}"
)


print("\nDelta:")
print(
    f"NDCG@10:   "
    f"{learned_metrics['ndcg@10'] - baseline_metrics['ndcg@10']:+.6f}"
)

print(
    f"MRR:       "
    f"{learned_metrics['mrr'] - baseline_metrics['mrr']:+.6f}"
)

print(
    f"Recall@10: "
    f"{learned_metrics['recall@10'] - baseline_metrics['recall@10']:+.6f}"
)


# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------

model_path = MODEL_DIR / "adaptive_fusion.joblib"

joblib.dump(
    {
        "model": final_model,
        "beta": chosen_beta,
        "feature_names": FEATURE_NAMES,
        "top_k": TOP_K,
        "dense_weights": {
            "embeddinggemma": 0.65,
            "qwen": 0.35,
        },
    },
    model_path,
)


result = {
    "training_queries": len(train_groups),
    "training_recoverable":
        len(recoverable_train),
    "validation_queries": len(val_groups),
    "validation_recoverable":
        len(val_groups) - val_missing,
    "inner_results": inner_results,
    "chosen_beta": chosen_beta,
    "dense_baseline": baseline_metrics,
    "adaptive_fusion": learned_metrics,
}


with open(
    RESULTS_DIR / "adaptive_fusion.json",
    "w",
) as f:
    json.dump(
        result,
        f,
        indent=2,
    )


print(
    "\nSaved model to "
    "models/adaptive_fusion.joblib"
)

print(
    "Saved metrics to "
    "results/adaptive_fusion.json"
)

print(
    "\n✅ Adaptive fusion experiment completed."
)
