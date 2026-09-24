from src.runtime_device import get_device

import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.data_loader import load_training_benchmark


print("=" * 78)
print("TRAINING QUERY EMBEDDING CACHE")
print("=" * 78)

queries, _, _ = load_training_benchmark()

query_ids = [
    row["_id"]
    for row in queries
]

query_texts = [
    row["text"]
    for row in queries
]

print(f"\nTraining queries: {len(queries)}")


# --------------------------------------------------
# SAVE QUERY ORDER
# --------------------------------------------------

id_path = Path(
    "cache/training_query_ids.json"
)

id_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(id_path, "w") as f:
    json.dump(query_ids, f)

print(
    "Saved query ordering to "
    "cache/training_query_ids.json"
)


# --------------------------------------------------
# EMBEDDINGGEMMA
# --------------------------------------------------

gemma_dir = Path(
    "cache/embeddinggemma"
)

gemma_dir.mkdir(
    parents=True,
    exist_ok=True
)

gemma_path = (
    gemma_dir
    / "training_queries.npy"
)


if gemma_path.exists():
    print(
        "\nEmbeddingGemma training "
        "embeddings already cached."
    )

    gemma_embeddings = np.load(
        gemma_path
    )

else:
    print(
        "\nLoading EmbeddingGemma..."
    )

    model = SentenceTransformer(
        "google/embeddinggemma-300m",
        device=get_device(),
        model_kwargs={
            "torch_dtype": torch.float32
        }
    )

    model.max_seq_length = 2048

    print(
        "Encoding 4,000 training queries "
        "with EmbeddingGemma..."
    )

    start = time.perf_counter()

    gemma_embeddings = model.encode_query(
        query_texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    np.save(
        gemma_path,
        gemma_embeddings
    )

    print(
        f"EmbeddingGemma completed in "
        f"{elapsed / 60:.2f} min"
    )

    del model


print(
    "EmbeddingGemma matrix:",
    gemma_embeddings.shape
)


# --------------------------------------------------
# QWEN
# --------------------------------------------------

qwen_dir = Path(
    "cache/qwen06b"
)

qwen_dir.mkdir(
    parents=True,
    exist_ok=True
)

qwen_path = (
    qwen_dir
    / "training_queries.npy"
)


if qwen_path.exists():
    print(
        "\nQwen training embeddings "
        "already cached."
    )

    qwen_embeddings = np.load(
        qwen_path
    )

else:
    print(
        "\nLoading Qwen..."
    )

    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        device=get_device()
    )

    model.max_seq_length = 4096

    print(
        "Encoding 4,000 training queries "
        "with Qwen..."
    )

    start = time.perf_counter()

    qwen_embeddings = model.encode_query(
        query_texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    np.save(
        qwen_path,
        qwen_embeddings
    )

    print(
        f"Qwen completed in "
        f"{elapsed / 60:.2f} min"
    )


print(
    "Qwen matrix:",
    qwen_embeddings.shape
)


# --------------------------------------------------
# SANITY CHECK
# --------------------------------------------------

assert gemma_embeddings.shape[0] == 4000
assert qwen_embeddings.shape[0] == 4000

assert gemma_embeddings.shape[1] == 768
assert qwen_embeddings.shape[1] == 1024


print("\n" + "=" * 78)
print("CACHE COMPLETE")
print("=" * 78)

print(
    "EmbeddingGemma:",
    gemma_path
)

print(
    "Qwen:",
    qwen_path
)

print(
    "Query IDs:",
    id_path
)

print(
    "\n✅ Training-query embeddings cached."
)
