import json

from datasets import load_dataset


def _load_raw_train_data():
    with open("data/split.json") as f:
        split = json.load(f)

    queries_ds = load_dataset(
        "CoIR-Retrieval/apps",
        "queries",
        split="queries"
    )

    corpus_ds = load_dataset(
        "CoIR-Retrieval/apps",
        "corpus",
        split="corpus"
    )

    train_corpus = [
        row
        for row in corpus_ds
        if row["partition"] == "train"
    ]

    train_queries = [
        row
        for row in queries_ds
        if row["partition"] == "train"
    ]

    return split, train_queries, train_corpus


def load_dev_benchmark():
    split, all_train_queries, train_corpus = _load_raw_train_data()

    validation_pairs = split["validation"]

    validation_query_ids = {
        x["query_id"]
        for x in validation_pairs
    }

    validation_queries = [
        row
        for row in all_train_queries
        if row["_id"] in validation_query_ids
    ]

    relevant_docs = {
        x["query_id"]: x["doc_id"]
        for x in validation_pairs
    }

    return (
        validation_queries,
        train_corpus,
        relevant_docs
    )


def load_training_benchmark():
    split, all_train_queries, train_corpus = _load_raw_train_data()

    training_pairs = split["train"]

    training_query_ids = {
        x["query_id"]
        for x in training_pairs
    }

    training_queries = [
        row
        for row in all_train_queries
        if row["_id"] in training_query_ids
    ]

    relevant_docs = {
        x["query_id"]: x["doc_id"]
        for x in training_pairs
    }

    return (
        training_queries,
        train_corpus,
        relevant_docs
    )
