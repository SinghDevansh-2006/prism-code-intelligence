import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.code_normalizer import compact_large_literals


BATCH_SIZE = 8

OFFICIAL_DIR = Path("cache/official")
OFFICIAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD RAW DATA
# ============================================================

print("=" * 80)
print("OFFICIAL APPSRETRIEVAL EMBEDDING PREPARATION")
print("=" * 80)

print("\nLoading Apps corpus + queries...")

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


test_docs = [
    row
    for row in corpus
    if row["partition"] == "test"
]

test_queries = [
    row
    for row in queries
    if row["partition"] == "test"
]


assert len(corpus) == 8765
assert len(test_docs) == 3765
assert len(test_queries) == 3765

assert corpus[0]["_id"] == "d1"
assert corpus[4999]["_id"] == "d5000"
assert test_docs[0]["_id"] == "d5001"
assert test_docs[-1]["_id"] == "d8765"


full_doc_ids = [
    row["_id"]
    for row in corpus
]

test_doc_ids = [
    row["_id"]
    for row in test_docs
]

test_query_ids = [
    row["_id"]
    for row in test_queries
]


with open(
    OFFICIAL_DIR / "document_ids.json",
    "w",
) as f:
    json.dump(
        full_doc_ids,
        f,
        indent=2,
    )


with open(
    OFFICIAL_DIR / "test_document_ids.json",
    "w",
) as f:
    json.dump(
        test_doc_ids,
        f,
        indent=2,
    )


with open(
    OFFICIAL_DIR / "query_ids.json",
    "w",
) as f:
    json.dump(
        test_query_ids,
        f,
        indent=2,
    )


print(
    f"Full corpus:          {len(corpus)}"
)

print(
    f"New documents:        {len(test_docs)}"
)

print(
    f"Official test queries:{len(test_queries)}"
)


# ============================================================
# PREPROCESS NEW DOCUMENTS ONLY
# ============================================================

print("\nCompacting literals in d5001..d8765...")

start = time.perf_counter()

test_doc_texts = [
    compact_large_literals(
        row["text"]
    )
    for row in test_docs
]

preprocess_seconds = (
    time.perf_counter()
    - start
)

query_texts = [
    row["text"]
    for row in test_queries
]


print(
    f"Preprocessing completed in "
    f"{preprocess_seconds:.2f} seconds."
)


# ============================================================
# HELPERS
# ============================================================

def free_model(model):
    del model
    gc.collect()

    if (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
    ):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def verify_existing_cache(
    path,
    expected_shape,
):
    array = np.load(
        path,
        mmap_mode="r",
    )

    if array.shape != expected_shape:
        raise RuntimeError(
            f"{path} expected shape "
            f"{expected_shape}, "
            f"found {array.shape}"
        )

    return array


# ============================================================
# EMBEDDINGGEMMA
# ============================================================

print("\n" + "=" * 80)
print("EMBEDDINGGEMMA")
print("=" * 80)


existing_gemma = verify_existing_cache(
    "cache/embeddinggemma/documents.npy",
    (5000, 768),
)


gemma_new_docs_path = (
    OFFICIAL_DIR
    / "embeddinggemma_test_documents.npy"
)

gemma_queries_path = (
    OFFICIAL_DIR
    / "embeddinggemma_queries.npy"
)

gemma_full_docs_path = (
    OFFICIAL_DIR
    / "embeddinggemma_documents.npy"
)


print("\nLoading EmbeddingGemma...")

gemma = SentenceTransformer(
    "google/embeddinggemma-300m",
    device="mps",
    model_kwargs={
        "torch_dtype": torch.float32
    },
)

gemma.max_seq_length = 2048


# ------------------------------------------------------------
# NEW DOCUMENTS
# ------------------------------------------------------------

if gemma_new_docs_path.exists():
    print(
        "\nLoading cached d5001..d8765 "
        "EmbeddingGemma embeddings..."
    )

    gemma_new_docs = np.load(
        gemma_new_docs_path
    )

else:
    print(
        "\nEncoding 3,765 new documents "
        "with EmbeddingGemma..."
    )

    start = time.perf_counter()

    gemma_new_docs = (
        gemma.encode_document(
            test_doc_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        .astype(np.float32)
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    np.save(
        gemma_new_docs_path,
        gemma_new_docs,
    )

    print(
        f"New document encoding: "
        f"{elapsed / 60:.2f} minutes"
    )


assert gemma_new_docs.shape == (
    3765,
    768,
)


# ------------------------------------------------------------
# OFFICIAL QUERIES
# ------------------------------------------------------------

if gemma_queries_path.exists():
    print(
        "\nLoading cached official "
        "EmbeddingGemma query embeddings..."
    )

    gemma_queries = np.load(
        gemma_queries_path
    )

else:
    print(
        "\nEncoding 3,765 official queries "
        "with EmbeddingGemma..."
    )

    start = time.perf_counter()

    gemma_queries = (
        gemma.encode_query(
            query_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        .astype(np.float32)
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    np.save(
        gemma_queries_path,
        gemma_queries,
    )

    print(
        f"Query encoding: "
        f"{elapsed / 60:.2f} minutes"
    )


assert gemma_queries.shape == (
    3765,
    768,
)


# ------------------------------------------------------------
# FULL 8,765 MATRIX
# ------------------------------------------------------------

if gemma_full_docs_path.exists():
    print(
        "\nFull EmbeddingGemma document "
        "matrix already exists."
    )

else:
    print(
        "\nCreating full 8,765-document "
        "EmbeddingGemma matrix..."
    )

    gemma_full = np.concatenate(
        [
            np.asarray(
                existing_gemma,
                dtype=np.float32,
            ),
            gemma_new_docs,
        ],
        axis=0,
    )

    assert gemma_full.shape == (
        8765,
        768,
    )

    np.save(
        gemma_full_docs_path,
        gemma_full,
    )


print(
    "EmbeddingGemma full docs:",
    np.load(
        gemma_full_docs_path,
        mmap_mode="r",
    ).shape,
)

print(
    "EmbeddingGemma queries:",
    gemma_queries.shape,
)


free_model(gemma)


# ============================================================
# QWEN
# ============================================================

print("\n" + "=" * 80)
print("QWEN3-EMBEDDING-0.6B")
print("=" * 80)


existing_qwen = verify_existing_cache(
    "cache/qwen06b/documents.npy",
    (5000, 1024),
)


qwen_new_docs_path = (
    OFFICIAL_DIR
    / "qwen_test_documents.npy"
)

qwen_queries_path = (
    OFFICIAL_DIR
    / "qwen_queries.npy"
)

qwen_full_docs_path = (
    OFFICIAL_DIR
    / "qwen_documents.npy"
)


print("\nLoading Qwen...")

qwen = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="mps",
)

qwen.max_seq_length = 4096


# ------------------------------------------------------------
# NEW DOCUMENTS
# ------------------------------------------------------------

if qwen_new_docs_path.exists():
    print(
        "\nLoading cached d5001..d8765 "
        "Qwen embeddings..."
    )

    qwen_new_docs = np.load(
        qwen_new_docs_path
    )

else:
    print(
        "\nEncoding 3,765 new documents "
        "with Qwen..."
    )

    start = time.perf_counter()

    qwen_new_docs = (
        qwen.encode_document(
            test_doc_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        .astype(np.float32)
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    np.save(
        qwen_new_docs_path,
        qwen_new_docs,
    )

    print(
        f"New document encoding: "
        f"{elapsed / 60:.2f} minutes"
    )


assert qwen_new_docs.shape == (
    3765,
    1024,
)


# ------------------------------------------------------------
# OFFICIAL QUERIES
# ------------------------------------------------------------

if qwen_queries_path.exists():
    print(
        "\nLoading cached official "
        "Qwen query embeddings..."
    )

    qwen_queries = np.load(
        qwen_queries_path
    )

else:
    print(
        "\nEncoding 3,765 official queries "
        "with Qwen..."
    )

    start = time.perf_counter()

    qwen_queries = (
        qwen.encode_query(
            query_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        .astype(np.float32)
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    np.save(
        qwen_queries_path,
        qwen_queries,
    )

    print(
        f"Query encoding: "
        f"{elapsed / 60:.2f} minutes"
    )


assert qwen_queries.shape == (
    3765,
    1024,
)


# ------------------------------------------------------------
# FULL 8,765 MATRIX
# ------------------------------------------------------------

if qwen_full_docs_path.exists():
    print(
        "\nFull Qwen document "
        "matrix already exists."
    )

else:
    print(
        "\nCreating full 8,765-document "
        "Qwen matrix..."
    )

    qwen_full = np.concatenate(
        [
            np.asarray(
                existing_qwen,
                dtype=np.float32,
            ),
            qwen_new_docs,
        ],
        axis=0,
    )

    assert qwen_full.shape == (
        8765,
        1024,
    )

    np.save(
        qwen_full_docs_path,
        qwen_full,
    )


print(
    "Qwen full docs:",
    np.load(
        qwen_full_docs_path,
        mmap_mode="r",
    ).shape,
)

print(
    "Qwen queries:",
    qwen_queries.shape,
)


free_model(qwen)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL OFFICIAL CACHE CHECK")
print("=" * 80)


checks = {
    "Gemma documents":
        (
            OFFICIAL_DIR
            / "embeddinggemma_documents.npy",
            (8765, 768),
        ),

    "Gemma queries":
        (
            OFFICIAL_DIR
            / "embeddinggemma_queries.npy",
            (3765, 768),
        ),

    "Qwen documents":
        (
            OFFICIAL_DIR
            / "qwen_documents.npy",
            (8765, 1024),
        ),

    "Qwen queries":
        (
            OFFICIAL_DIR
            / "qwen_queries.npy",
            (3765, 1024),
        ),
}


for name, (
    path,
    expected_shape,
) in checks.items():

    arr = np.load(
        path,
        mmap_mode="r",
    )

    print(
        f"{name:18}: "
        f"{arr.shape} "
        f"{arr.dtype}"
    )

    assert arr.shape == expected_shape
    assert arr.dtype == np.float32


print(
    "\nNo qrels or official metrics "
    "were used in this preparation step."
)

print(
    "\n✅ Official frozen embeddings are ready."
)
