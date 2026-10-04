"""Validate every submitted ID before independently recomputing retrieval metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QRELS_REVISION = "a4fb4d92996bcfe1e0b0af9e97ea4b70b80ec9d5"
CORPUS_REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_rankings(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def validate_rankings(rankings, qrels, corpus_ids, depth=100):
    if not isinstance(rankings, dict) or not qrels:
        raise ValueError("Expected ranking object and nonempty qrels")
    missing, extra = set(qrels) - set(rankings), set(rankings) - set(qrels)
    if missing or extra:
        raise ValueError(f"Query coverage: {len(missing)} missing, {len(extra)} unexpected")
    for qid, docs in rankings.items():
        if not isinstance(docs, list) or len(docs) != depth:
            raise ValueError(f"{qid}: expected {depth} ranked IDs")
        if not all(isinstance(doc, str) for doc in docs):
            raise ValueError(f"{qid}: document IDs must be strings")
        if len(set(docs)) != len(docs):
            raise ValueError(f"{qid}: duplicate document IDs")
        unknown = set(docs) - corpus_ids
        if unknown:
            raise ValueError(f"{qid}: unknown corpus IDs: {sorted(unknown)[:3]}")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, default=ROOT / "submission/AppsRetrieval_inference_results.json")
    parser.add_argument("--qrels-arrow", type=Path, help="Offline official test qrels Arrow")
    parser.add_argument("--corpus-arrow", type=Path, help="Offline official corpus Arrow")
    args = parser.parse_args()
    from datasets import Dataset, load_dataset
    qrows = Dataset.from_file(str(args.qrels_arrow)) if args.qrels_arrow else load_dataset(
        "CoIR-Retrieval/apps-qrels", revision=QRELS_REVISION, split="test")
    corpus = Dataset.from_file(str(args.corpus_arrow)) if args.corpus_arrow else load_dataset(
        "CoIR-Retrieval/apps", "corpus", revision=CORPUS_REVISION, split="corpus")
    qrels = {}
    for row in qrows:
        qrels.setdefault(str(row["query_id"]), {})[str(row["corpus_id"])] = float(row["score"])
    rankings = load_rankings(args.rankings)
    corpus_ids = set(corpus["_id"])
    if len(qrels) != 3765 or len(corpus_ids) != 8765:
        raise ValueError("Unexpected official dataset size")
    validate_rankings(rankings, qrels, corpus_ids)
    metrics = {}
    for name, fn, k in [("NDCG@10", ndcg_at_k, 10), ("MRR@10", reciprocal_rank, 10),
                         ("MRR@100", reciprocal_rank, 100), ("Recall@10", recall_at_k, 10),
                         ("Recall@20", recall_at_k, 20), ("Recall@100", recall_at_k, 100)]:
        metrics[name] = sum(fn(rankings[q], rel, k) for q, rel in qrels.items()) / len(qrels)
    print(json.dumps({"sha256": hashlib.sha256(args.rankings.read_bytes()).hexdigest(),
        "queries": len(rankings), "corpus_documents": len(corpus_ids),
        "validation": "Exact query coverage; 100 unique valid IDs per query", "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
