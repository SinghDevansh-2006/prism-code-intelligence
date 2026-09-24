import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


RUN_PATH = Path(
    "submission/AppsRetrieval_inference_results.json"
)


# ------------------------------------------------------------
# LOAD SUBMITTED RANKINGS
# ------------------------------------------------------------

with open(RUN_PATH) as f:
    rankings = json.load(f)


# ------------------------------------------------------------
# ARTIFACT CHECKSUM
# ------------------------------------------------------------

sha256 = hashlib.sha256(
    RUN_PATH.read_bytes()
).hexdigest()


# ------------------------------------------------------------
# LOAD OFFICIAL APPS QRELS
# ------------------------------------------------------------

print("Loading official AppsRetrieval test qrels...")

qrels_dataset = load_dataset(
    "CoIR-Retrieval/apps-qrels",
    split="test",
)


qrels = defaultdict(dict)

for row in qrels_dataset:
    qrels[str(row["query_id"])][
        str(row["corpus_id"])
    ] = float(row["score"])


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def dcg_at_k(ranking, relevant, k):
    total = 0.0

    for rank, doc_id in enumerate(
        ranking[:k],
        start=1,
    ):
        rel = relevant.get(
            doc_id,
            0.0,
        )

        total += (
            (2.0 ** rel - 1.0)
            / math.log2(rank + 1)
        )

    return total


def ndcg_at_k(ranking, relevant, k):
    dcg = dcg_at_k(
        ranking,
        relevant,
        k,
    )

    ideal_rels = sorted(
        relevant.values(),
        reverse=True,
    )[:k]

    idcg = sum(
        (
            (2.0 ** rel - 1.0)
            / math.log2(rank + 1)
        )
        for rank, rel in enumerate(
            ideal_rels,
            start=1,
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def reciprocal_rank(ranking, relevant, k):
    for rank, doc_id in enumerate(
        ranking[:k],
        start=1,
    ):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank

    return 0.0


def recall_at_k(ranking, relevant, k):
    relevant_ids = {
        doc_id
        for doc_id, score in relevant.items()
        if score > 0
    }

    if not relevant_ids:
        return 0.0

    retrieved = set(
        ranking[:k]
    )

    return (
        len(
            relevant_ids
            & retrieved
        )
        / len(relevant_ids)
    )


ndcg10 = []
mrr10 = []
mrr100 = []
recall10 = []
recall20 = []
recall100 = []

missing = []


for query_id, relevant in qrels.items():

    if query_id not in rankings:
        missing.append(query_id)
        continue

    ranking = rankings[query_id]

    ndcg10.append(
        ndcg_at_k(
            ranking,
            relevant,
            10,
        )
    )

    mrr10.append(
        reciprocal_rank(
            ranking,
            relevant,
            10,
        )
    )

    mrr100.append(
        reciprocal_rank(
            ranking,
            relevant,
            100,
        )
    )

    recall10.append(
        recall_at_k(
            ranking,
            relevant,
            10,
        )
    )

    recall20.append(
        recall_at_k(
            ranking,
            relevant,
            20,
        )
    )

    recall100.append(
        recall_at_k(
            ranking,
            relevant,
            100,
        )
    )


if missing:
    raise RuntimeError(
        f"Missing {len(missing)} queries "
        f"from submission rankings."
    )


def mean(values):
    return sum(values) / len(values)


print()
print("=" * 80)
print("SUBMISSION ARTIFACT VERIFICATION")
print("=" * 80)

print(
    "Artifact:",
    RUN_PATH,
)

print(
    "SHA256:",
    sha256,
)

print(
    "Queries:",
    len(rankings),
)

print(
    "Qrels queries:",
    len(qrels),
)

print()
print("Metrics computed directly from submitted ranking order:")
print()

print(
    f"NDCG@10    : {mean(ndcg10):.6f}"
)

print(
    f"MRR@10     : {mean(mrr10):.6f}"
)

print(
    f"MRR@100    : {mean(mrr100):.6f}"
)

print(
    f"Recall@10  : {mean(recall10):.6f}"
)

print(
    f"Recall@20  : {mean(recall20):.6f}"
)

print(
    f"Recall@100 : {mean(recall100):.6f}"
)

print()
print(
    "✅ Metrics independently recomputed "
    "from the submitted top-100 rankings."
)
