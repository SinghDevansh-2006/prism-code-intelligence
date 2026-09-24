import time
from pathlib import Path

from src.query_router import route_query
from src.retriever import CodeRetriever
from src.structural_search import StructuralSearchEngine
from src.versioned_retriever import VersionedSemanticIndex


class AgenticCodeEngine:
    def __init__(
        self,
        index_dir="runtime_index",
        device="mps",
        enable_reranker=True,
        version_index_root=None,
    ):
        self.index_dir = index_dir
        self.device = device
        self.enable_reranker = enable_reranker

        self.version_index_root = (
            version_index_root
        )

        # Everything is lazy-loaded.
        self.semantic_engine = None
        self.structural_engine = None
        self.version_engine = None

    # ======================================================
    # LAZY ENGINE LOADERS
    # ======================================================

    def _get_semantic_engine(self):
        if self.semantic_engine is None:
            print(
                "Initializing semantic retriever..."
            )

            shared_gemma = None

            if self.version_engine is not None:
                shared_gemma = (
                    self.version_engine.model
                )

            self.semantic_engine = CodeRetriever(
                index_dir=self.index_dir,
                device=self.device,
                enable_reranker=
                    self.enable_reranker,
                gemma_model=
                    shared_gemma,
            )

        return self.semantic_engine

    def _get_structural_engine(self):
        if self.structural_engine is None:
            print(
                "Initializing structural engine..."
            )

            self.structural_engine = (
                StructuralSearchEngine(
                    index_dir=self.index_dir
                )
            )

        return self.structural_engine

    def _get_version_engine(self):
        if self.version_engine is not None:
            return self.version_engine

        if self.version_index_root is None:
            return None

        root = Path(
            self.version_index_root
        )

        versions_path = (
            root / "versions.jsonl"
        )

        if not versions_path.exists():
            return None

        print(
            "Initializing evolutionary retriever..."
        )

        shared_gemma = None

        if self.semantic_engine is not None:
            shared_gemma = (
                self.semantic_engine.gemma
            )

        self.version_engine = (
            VersionedSemanticIndex(
                root=root,
                device=self.device,
                shared_model=
                    shared_gemma,
            )
        )

        return self.version_engine

    # ======================================================
    # SEARCH
    # ======================================================

    def search(
        self,
        query,
        top_k=10,
    ):
        total_start = time.perf_counter()

        plan = route_query(
            query
        )

        # --------------------------------------------------
        # SEMANTIC
        # --------------------------------------------------

        if plan.route == "semantic":
            semantic = (
                self._get_semantic_engine()
            )

            response = semantic.search(
                query,
                top_k=top_k,
            )

            results = response[
                "results"
            ]

            execution = {
                "strategy":
                    "dual_dense_retrieval",

                "steps": [
                    "Encode query with EmbeddingGemma",
                    "Encode query with Qwen3-Embedding",
                    "Normalize retrieval score distributions",
                    "Fuse scores using 65/35 weighting",
                    "Measure retrieval confidence",
                ],

                "reranker_triggered":
                    response[
                        "retrieval"
                    ][
                        "reranker_triggered"
                    ],

                "confidence_margin":
                    response[
                        "retrieval"
                    ][
                        "confidence_margin"
                    ],

                "confidence_threshold":
                    response[
                        "retrieval"
                    ][
                        "confidence_threshold"
                    ],

                "timings_ms":
                    response[
                        "retrieval"
                    ][
                        "timings_ms"
                    ],
            }

            if execution[
                "reranker_triggered"
            ]:
                execution[
                    "steps"
                ].append(
                    "Rerank top-20 candidates "
                    "with confidence-gated cross-encoder"
                )

        # --------------------------------------------------
        # EXACT USAGE
        # --------------------------------------------------

        elif plan.route == "exact_usage":
            structural = (
                self._get_structural_engine()
            )

            response = structural.search(
                query,
                top_k=top_k,
            )

            results = response[
                "results"
            ]

            execution = {
                "strategy":
                    "symbolic_usage_search",

                "steps": [
                    "Extract identifier/API terms",
                    "Search AST-derived usage index",
                    "Collect call/import/function evidence",
                    "Return line-aware matches",
                ],

                "terms":
                    plan.extracted_terms,
            }

        # --------------------------------------------------
        # STRUCTURAL
        # --------------------------------------------------

        elif plan.route == "structural":
            structural = (
                self._get_structural_engine()
            )

            response = structural.search(
                query,
                top_k=top_k,
            )

            results = response[
                "results"
            ]

            execution = {
                "strategy":
                    "ast_structural_search",

                "steps": [
                    "Parse structural constraints",
                    "Search scope-aware AST metadata",
                    "Verify function/class/call-order constraints",
                    "Return structural evidence",
                ],

                "terms":
                    plan.extracted_terms,
            }

        # --------------------------------------------------
        # EVOLUTION
        # --------------------------------------------------

        elif plan.route == "evolution":
            version_engine = (
                self._get_version_engine()
            )

            if version_engine is None:
                results = []

                execution = {
                    "strategy":
                        "evolutionary_retrieval",

                    "available":
                        False,

                    "steps": [
                        "Detected version-history query",
                        "No version repository is currently attached",
                    ],
                }

            else:
                response = (
                    version_engine.search(
                        query,
                        top_k=top_k,
                    )
                )

                results = []

                for result in response[
                    "results"
                ]:
                    history = (
                        version_engine.history(
                            result[
                                "logical_id"
                            ]
                        )
                    )

                    timeline = [
                        {
                            "version":
                                item[
                                    "version"
                                ],

                            "change_type":
                                item[
                                    "change_type"
                                ],

                            "previous_similarity":
                                item[
                                    "previous_similarity"
                                ],
                        }
                        for item in history
                    ]

                    results.append(
                        {
                            **result,
                            "timeline":
                                timeline,
                        }
                    )

                execution = {
                    "strategy":
                        "evolutionary_retrieval",

                    "available":
                        True,

                    "steps": [
                        "Route query to version-aware index",
                        "Embed query for semantic version retrieval",
                        "Search across every indexed version",
                        "Resolve matching versions to lineage",
                        "Attach change classification and history",
                    ],

                    "timings_ms":
                        response[
                            "timings_ms"
                        ],
                }

        else:
            raise ValueError(
                f"Unknown route: {plan.route}"
            )

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        return {
            "query":
                query,

            "plan": {
                "route":
                    plan.route,

                "confidence":
                    plan.confidence,

                "reason":
                    plan.reason,

                "extracted_terms":
                    plan.extracted_terms,
            },

            "execution":
                execution,

            "results":
                results,

            "total_ms":
                total_ms,
        }
