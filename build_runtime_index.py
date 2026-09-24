import json
from pathlib import Path

import numpy as np

from src.data_loader import load_training_benchmark
from src.code_normalizer import compact_large_literals


OUT_DIR = Path("runtime_index")
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("BUILDING LOCAL RUNTIME INDEX")
print("=" * 80)


# --------------------------------------------------
# LOAD CORPUS
# --------------------------------------------------

_, corpus, _ = load_training_benchmark()

doc_ids = [
    row["_id"]
    for row in corpus
]


print(f"\nCorpus documents: {len(corpus)}")


# --------------------------------------------------
# VERIFY EMBEDDING ALIGNMENT
# --------------------------------------------------

qwen_docs = np.load(
    "cache/qwen06b/documents.npy",
    mmap_mode="r"
)

gemma_docs = np.load(
    "cache/embeddinggemma/documents.npy",
    mmap_mode="r"
)


assert len(corpus) == qwen_docs.shape[0]
assert len(corpus) == gemma_docs.shape[0]

assert qwen_docs.shape[1] == 1024
assert gemma_docs.shape[1] == 768


print(
    "Qwen document matrix:",
    qwen_docs.shape
)

print(
    "Gemma document matrix:",
    gemma_docs.shape
)


# --------------------------------------------------
# SAVE CORPUS
# --------------------------------------------------

corpus_path = (
    OUT_DIR
    / "corpus.jsonl"
)

print(
    "\nWriting local corpus..."
)

with open(
    corpus_path,
    "w"
) as f:

    for row in corpus:
        record = {
            "id": row["_id"],
            "title": row.get(
                "title",
                ""
            ),
            "language": row.get(
                "language",
                ""
            ),
            "code": row["text"],
            "retrieval_code":
                compact_large_literals(
                    row["text"]
                ),
        }

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# --------------------------------------------------
# SAVE DOCUMENT ORDER
# --------------------------------------------------

with open(
    OUT_DIR
    / "document_ids.json",
    "w"
) as f:
    json.dump(
        doc_ids,
        f,
        indent=2
    )


# --------------------------------------------------
# COPY FROZEN RETRIEVAL CONFIG
# --------------------------------------------------

with open(
    "results/confidence_threshold.json"
) as f:
    threshold_data = json.load(f)

with open(
    "results/reranker_beta_training.json"
) as f:
    beta_data = json.load(f)


config = {
    "version": "1.0",
    "corpus_size": len(corpus),

    "dense_retrieval": {
        "embeddinggemma_weight": 0.65,
        "qwen_weight": 0.35,

        "embeddinggemma_model":
            "google/embeddinggemma-300m",

        "qwen_model":
            "Qwen/Qwen3-Embedding-0.6B",
    },

    "confidence_gate": {
        "type": "top1_top2_margin",

        "threshold":
            threshold_data[
                "margin_threshold"
            ],

        "threshold_source":
            "4000 training queries",
    },

    "reranking": {
        "enabled": True,

        "model":
            "mixedbread-ai/"
            "mxbai-rerank-xsmall-v1",

        "top_k": 20,

        "beta":
            beta_data[
                "chosen"
            ]["beta"],

        "beta_source":
            "training-only "
            "low-confidence sample",
    },

    "output_top_k": 10,

    "validation": {
        "ndcg@10": 0.920441,
        "mrr": 0.903767,
        "recall@10": 0.974000,
        "recall@20": 0.984000,
        "recall@100": 0.993000,
    },
}


with open(
    OUT_DIR
    / "config.json",
    "w"
) as f:
    json.dump(
        config,
        f,
        indent=2
    )


# --------------------------------------------------
# VERIFY LOCAL CORPUS
# --------------------------------------------------

with open(
    corpus_path
) as f:
    lines = f.readlines()


assert len(lines) == 5000

first = json.loads(
    lines[0]
)

last = json.loads(
    lines[-1]
)


print("\n" + "=" * 80)
print("RUNTIME INDEX READY")
print("=" * 80)

print(
    "Corpus:",
    corpus_path
)

print(
    "Document IDs:",
    OUT_DIR / "document_ids.json"
)

print(
    "Config:",
    OUT_DIR / "config.json"
)

print(
    "\nFirst document:",
    first["id"]
)

print(
    "Last document:",
    last["id"]
)

print(
    "Documents written:",
    len(lines)
)

print(
    "\nFrozen NDCG@10:",
    config["validation"]["ndcg@10"]
)

print(
    "Frozen MRR:",
    config["validation"]["mrr"]
)

print(
    "\n✅ Local runtime index built successfully."
)
