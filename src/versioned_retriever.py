import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.version_index import (
    exact_fingerprint,
    token_set,
    jaccard_similarity,
)


EMBEDDING_DIM = 768


class VersionedSemanticIndex:
    """
    Append-only semantic index for code versions.

    Storage:
      versions.jsonl   -> one metadata record per version
      embeddings.f32   -> raw contiguous float32 vectors

    Adding one version appends one JSON line and one embedding.
    Existing embeddings are never rebuilt or rewritten.
    """

    def __init__(
        self,
        root="runtime_index/versioned_semantic",
        device="cpu",
        similarity_threshold=0.75,
        shared_model=None,
    ):
        self.root = Path(root)

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.versions_path = (
            self.root / "versions.jsonl"
        )

        self.embeddings_path = (
            self.root / "embeddings.f32"
        )

        self.similarity_threshold = (
            similarity_threshold
        )

        self.records = []
        self.lineages = defaultdict(list)
        self.existing_keys = set()

        self._load_metadata()

        if shared_model is None:
            print(
                "Loading version retrieval model..."
            )

            self.model = SentenceTransformer(
                "google/embeddinggemma-300m",
                device=device,
                model_kwargs={
                    "torch_dtype": torch.float32
                },
            )

            self.model.max_seq_length = 2048

        else:
            print(
                "Reusing shared EmbeddingGemma "
                "for version retrieval..."
            )

            self.model = shared_model

    # --------------------------------------------------
    # LOAD EXISTING APPEND-ONLY METADATA
    # --------------------------------------------------

    def _load_metadata(self):
        if not self.versions_path.exists():
            return

        with open(
            self.versions_path
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(
                    line
                )

                self.records.append(
                    record
                )

                self.lineages[
                    record["logical_id"]
                ].append(
                    record
                )

                self.existing_keys.add(
                    (
                        str(
                            record[
                                "logical_id"
                            ]
                        ),
                        str(
                            record[
                                "version"
                            ]
                        ),
                    )
                )

        expected_bytes = (
            len(self.records)
            * EMBEDDING_DIM
            * np.dtype(
                np.float32
            ).itemsize
        )

        if self.embeddings_path.exists():
            actual_bytes = (
                self.embeddings_path
                .stat()
                .st_size
            )

            if (
                actual_bytes
                != expected_bytes
            ):
                raise RuntimeError(
                    "Version metadata and "
                    "embedding file are inconsistent. "
                    f"Expected {expected_bytes} bytes, "
                    f"found {actual_bytes}."
                )

        elif self.records:
            raise RuntimeError(
                "Version metadata exists but "
                "embedding file is missing."
            )

    # --------------------------------------------------
    # EMBEDDING MATRIX VIEW
    # --------------------------------------------------

    def _embedding_matrix(self):
        if not self.records:
            return None

        return np.memmap(
            self.embeddings_path,
            dtype=np.float32,
            mode="r",
            shape=(
                len(self.records),
                EMBEDDING_DIM,
            ),
        )

    # --------------------------------------------------
    # APPEND ONE EMBEDDING
    # --------------------------------------------------

    def _append_embedding(
        self,
        embedding,
    ):
        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.shape != (
            EMBEDDING_DIM,
        ):
            raise ValueError(
                f"Expected embedding shape "
                f"({EMBEDDING_DIM},), "
                f"got {embedding.shape}"
            )

        with open(
            self.embeddings_path,
            "ab"
        ) as f:
            embedding.tofile(f)

    # --------------------------------------------------
    # APPEND ONE VERSION RECORD
    # --------------------------------------------------

    def _append_record(
        self,
        record,
    ):
        with open(
            self.versions_path,
            "a"
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------
    # ADD VERSION
    # --------------------------------------------------

    def add_version(
        self,
        logical_id,
        version,
        code,
        title="",
        language="python",
        metadata=None,
    ):
        total_start = time.perf_counter()

        key = (
            str(logical_id),
            str(version),
        )

        if key in self.existing_keys:
            raise ValueError(
                f"Version {version} already "
                f"exists for {logical_id}"
            )

        # ----------------------------------------------
        # LINEAGE / CHANGE DETECTION
        # ----------------------------------------------

        lineage_start = time.perf_counter()

        fingerprint = exact_fingerprint(
            code
        )

        lineage = self.lineages.get(
            logical_id,
            []
        )

        if not lineage:
            similarity = None
            change_type = "initial"

        else:
            previous = lineage[-1]

            if (
                previous["fingerprint"]
                == fingerprint
            ):
                similarity = 1.0
                change_type = "unchanged"

            else:
                similarity = (
                    jaccard_similarity(
                        token_set(
                            previous["code"]
                        ),
                        token_set(code),
                    )
                )

                if (
                    similarity
                    >= self.similarity_threshold
                ):
                    change_type = (
                        "near_duplicate_update"
                    )

                else:
                    change_type = (
                        "substantial_update"
                    )

        lineage_ms = (
            time.perf_counter()
            - lineage_start
        ) * 1000

        # ----------------------------------------------
        # EMBED ONLY THE NEW VERSION
        # ----------------------------------------------

        embedding_start = time.perf_counter()

        embedding = (
            self.model.encode_document(
                [code],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            .astype(
                np.float32
            )
        )

        embedding_ms = (
            time.perf_counter()
            - embedding_start
        ) * 1000

        # ----------------------------------------------
        # RECORD
        # ----------------------------------------------

        record = {
            "logical_id":
                logical_id,

            "version":
                version,

            "title":
                title,

            "language":
                language,

            "metadata":
                metadata or {},

            "code":
                code,

            "fingerprint":
                fingerprint,

            "previous_similarity":
                similarity,

            "change_type":
                change_type,
        }

        # ----------------------------------------------
        # APPEND ONLY
        # ----------------------------------------------

        persistence_start = (
            time.perf_counter()
        )

        # Embedding first; metadata second.
        self._append_embedding(
            embedding
        )

        self._append_record(
            record
        )

        persistence_ms = (
            time.perf_counter()
            - persistence_start
        ) * 1000

        # ----------------------------------------------
        # UPDATE IN-MEMORY STATE
        # ----------------------------------------------

        self.records.append(
            record
        )

        self.lineages[
            logical_id
        ].append(
            record
        )

        self.existing_keys.add(
            key
        )

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        return {
            "change_type":
                change_type,

            "previous_similarity":
                similarity,

            "lineage_index_ms":
                lineage_ms,

            "embedding_ms":
                embedding_ms,

            "persistence_ms":
                persistence_ms,

            "total_update_ms":
                total_ms,

            "total_versions":
                len(self.records),
        }

    # --------------------------------------------------
    # VERSION HISTORY
    # --------------------------------------------------

    def history(
        self,
        logical_id,
    ):
        return self.lineages.get(
            logical_id,
            []
        )

    # --------------------------------------------------
    # SEARCH ACROSS ALL VERSIONS
    # --------------------------------------------------

    def search(
        self,
        query,
        top_k=10,
    ):
        if not self.records:
            return {
                "query": query,
                "results": [],
                "timings_ms": {
                    "query_embedding": 0.0,
                    "vector_search": 0.0,
                    "total": 0.0,
                },
            }

        embedding_start = (
            time.perf_counter()
        )

        query_embedding = (
            self.model.encode_query(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            .astype(
                np.float32
            )
        )

        query_ms = (
            time.perf_counter()
            - embedding_start
        ) * 1000

        search_start = (
            time.perf_counter()
        )

        matrix = (
            self._embedding_matrix()
        )

        scores = (
            query_embedding
            @ matrix.T
        )

        result_count = min(
            top_k,
            len(self.records),
        )

        if (
            result_count
            == len(self.records)
        ):
            order = np.argsort(
                -scores
            )

        else:
            order = np.argpartition(
                -scores,
                result_count - 1,
            )[
                :result_count
            ]

            order = order[
                np.argsort(
                    -scores[order]
                )
            ]

        search_ms = (
            time.perf_counter()
            - search_start
        ) * 1000

        results = []

        for rank, idx in enumerate(
            order[:result_count],
            start=1,
        ):
            record = self.records[
                int(idx)
            ]

            results.append(
                {
                    "rank":
                        rank,

                    "logical_id":
                        record[
                            "logical_id"
                        ],

                    "version":
                        record[
                            "version"
                        ],

                    "score":
                        float(
                            scores[idx]
                        ),

                    "change_type":
                        record[
                            "change_type"
                        ],

                    "previous_similarity":
                        record[
                            "previous_similarity"
                        ],

                    "code":
                        record[
                            "code"
                        ],
                }
            )

        return {
            "query":
                query,

            "results":
                results,

            "timings_ms": {
                "query_embedding":
                    query_ms,

                "vector_search":
                    search_ms,

                "total":
                    query_ms
                    + search_ms,
            },
        }

    # --------------------------------------------------
    # STATS
    # --------------------------------------------------

    def stats(self):
        file_bytes = 0

        if self.embeddings_path.exists():
            file_bytes = (
                self.embeddings_path
                .stat()
                .st_size
            )

        return {
            "logical_snippets":
                len(self.lineages),

            "versions":
                len(self.records),

            "embedding_file_bytes":
                file_bytes,

            "bytes_per_version":
                EMBEDDING_DIM
                * 4,
        }
