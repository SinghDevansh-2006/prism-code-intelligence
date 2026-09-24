import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


class CodeRetriever:
    def __init__(
        self,
        index_dir="runtime_index",
        device="mps",
        enable_reranker=True,
        gemma_model=None,
    ):
        self.index_dir = Path(index_dir)
        self.device = device
        self.enable_reranker = enable_reranker

        # --------------------------------------------------
        # CONFIG
        # --------------------------------------------------

        with open(
            self.index_dir / "config.json"
        ) as f:
            self.config = json.load(f)

        with open(
            self.index_dir / "document_ids.json"
        ) as f:
            self.doc_ids = json.load(f)

        # --------------------------------------------------
        # CORPUS
        # --------------------------------------------------

        self.corpus = []

        with open(
            self.index_dir / "corpus.jsonl"
        ) as f:
            for line in f:
                self.corpus.append(
                    json.loads(line)
                )

        assert len(self.corpus) == len(
            self.doc_ids
        )

        # --------------------------------------------------
        # DOCUMENT EMBEDDINGS
        # --------------------------------------------------

        self.gemma_docs = np.load(
            self.index_dir
            / "embeddinggemma_documents.npy",
            mmap_mode="r",
        )

        self.qwen_docs = np.load(
            self.index_dir
            / "qwen_documents.npy",
            mmap_mode="r",
        )

        assert self.gemma_docs.shape[0] == len(
            self.doc_ids
        )

        assert self.qwen_docs.shape[0] == len(
            self.doc_ids
        )

        # --------------------------------------------------
        # MODELS
        # --------------------------------------------------

        dense_cfg = self.config[
            "dense_retrieval"
        ]

        if gemma_model is None:
            print(
                "Loading EmbeddingGemma..."
            )

            self.gemma = SentenceTransformer(
                dense_cfg[
                    "embeddinggemma_model"
                ],
                device=device,
                model_kwargs={
                    "torch_dtype": torch.float32
                },
            )

            self.gemma.max_seq_length = 2048

        else:
            print(
                "Reusing shared EmbeddingGemma..."
            )

            self.gemma = gemma_model

        print("Loading Qwen...")

        self.qwen = SentenceTransformer(
            dense_cfg[
                "qwen_model"
            ],
            device=device,
        )

        self.qwen.max_seq_length = 4096

        # --------------------------------------------------
        # RERANKER
        # lazy-loaded only if actually needed
        # --------------------------------------------------

        self.reranker = None

    # ======================================================
    # HELPERS
    # ======================================================

    @staticmethod
    def _zscore(row):
        mean = row.mean()
        std = row.std() + 1e-8

        return (
            row - mean
        ) / std

    @staticmethod
    def _top_margin(row):
        top2 = np.partition(
            row,
            -2
        )[-2:]

        return float(
            np.max(top2)
            - np.min(top2)
        )

    def _load_reranker(self):
        if self.reranker is not None:
            return

        print(
            "Loading confidence-gated "
            "reranker..."
        )

        rerank_cfg = self.config[
            "reranking"
        ]

        self.reranker = CrossEncoder(
            rerank_cfg["model"],
            device=self.device,
            max_length=512,
        )

    # ======================================================
    # DENSE RETRIEVAL
    # ======================================================

    def _dense_search(
        self,
        query,
    ):
        start = time.perf_counter()

        gemma_query = self.gemma.encode_query(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        gemma_encode_ms = (
            time.perf_counter() - start
        ) * 1000

        start = time.perf_counter()

        qwen_query = self.qwen.encode_query(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        qwen_encode_ms = (
            time.perf_counter() - start
        ) * 1000

        start = time.perf_counter()

        gemma_scores = (
            gemma_query
            @ self.gemma_docs.T
        )

        qwen_scores = (
            qwen_query
            @ self.qwen_docs.T
        )

        gemma_scores = self._zscore(
            gemma_scores
        )

        qwen_scores = self._zscore(
            qwen_scores
        )

        dense_cfg = self.config[
            "dense_retrieval"
        ]

        scores = (
            dense_cfg[
                "embeddinggemma_weight"
            ]
            * gemma_scores
            +
            dense_cfg[
                "qwen_weight"
            ]
            * qwen_scores
        )

        search_ms = (
            time.perf_counter() - start
        ) * 1000

        return (
            scores,
            {
                "embeddinggemma_ms":
                    gemma_encode_ms,

                "qwen_ms":
                    qwen_encode_ms,

                "vector_search_ms":
                    search_ms,
            },
        )

    # ======================================================
    # RERANKING
    # ======================================================

    def _rerank(
        self,
        query,
        dense_scores,
        dense_order,
    ):
        rerank_cfg = self.config[
            "reranking"
        ]

        top_k = rerank_cfg[
            "top_k"
        ]

        beta = rerank_cfg[
            "beta"
        ]

        candidate_indices = (
            dense_order[:top_k]
        )

        pairs = [
            (
                query,
                self.corpus[idx][
                    "retrieval_code"
                ],
            )
            for idx in candidate_indices
        ]

        self._load_reranker()

        start = time.perf_counter()

        rerank_scores = np.asarray(
            self.reranker.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
            )
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        dense_top = dense_scores[
            candidate_indices
        ]

        dense_z = self._zscore(
            dense_top
        )

        rerank_z = self._zscore(
            rerank_scores
        )

        blended = (
            (1.0 - beta)
            * dense_z
            + beta
            * rerank_z
        )

        new_positions = np.argsort(
            -blended
        )

        reranked_top = (
            candidate_indices[
                new_positions
            ]
        )

        final_order = np.concatenate(
            [
                reranked_top,
                dense_order[top_k:],
            ]
        )

        return (
            final_order,
            elapsed_ms,
        )

    # ======================================================
    # PUBLIC SEARCH API
    # ======================================================

    def search(
        self,
        query,
        top_k=10,
    ):
        total_start = time.perf_counter()

        dense_scores, timings = (
            self._dense_search(query)
        )

        dense_order = np.argsort(
            -dense_scores
        )

        margin = self._top_margin(
            dense_scores
        )

        threshold = self.config[
            "confidence_gate"
        ]["threshold"]

        gated = (
            self.enable_reranker
            and margin <= threshold
        )

        rerank_ms = 0.0

        if gated:
            final_order, rerank_ms = (
                self._rerank(
                    query,
                    dense_scores,
                    dense_order,
                )
            )

        else:
            final_order = dense_order

        results = []

        for rank, idx in enumerate(
            final_order[:top_k],
            start=1,
        ):
            item = self.corpus[idx]

            results.append(
                {
                    "rank": rank,
                    "id": item["id"],
                    "title": item["title"],
                    "language":
                        item["language"],
                    "score":
                        float(
                            dense_scores[idx]
                        ),
                    "code":
                        item["code"],
                }
            )

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        timings[
            "reranker_ms"
        ] = rerank_ms

        timings[
            "total_ms"
        ] = total_ms

        return {
            "query": query,
            "results": results,

            "retrieval": {
                "dense_models": [
                    "EmbeddingGemma",
                    "Qwen3-Embedding-0.6B",
                ],

                "confidence_margin":
                    margin,

                "confidence_threshold":
                    threshold,

                "reranker_triggered":
                    gated,

                "timings_ms":
                    timings,
            },
        }
