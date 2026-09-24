import json
import re
from pathlib import Path


class EvolutionSearchEngine:
    def __init__(
        self,
        index_path="runtime_index/version_index_demo.json",
    ):
        index_path = Path(index_path)

        with open(index_path) as f:
            payload = json.load(f)

        self.lineages = payload[
            "lineages"
        ]

        self.stats = payload[
            "stats"
        ]

    # --------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------

    @staticmethod
    def _normalize(text):
        return re.sub(
            r"[^a-z0-9]+",
            " ",
            text.lower()
        ).strip()

    # --------------------------------------------------
    # LINEAGE MATCHING
    # --------------------------------------------------

    def _score_lineage(
        self,
        logical_id,
        versions,
        query,
    ):
        query_words = set(
            self._normalize(query).split()
        )

        logical_words = set(
            self._normalize(
                logical_id
            ).split()
        )

        title_words = set()

        for version in versions:
            title_words.update(
                self._normalize(
                    version.get(
                        "title",
                        ""
                    )
                ).split()
            )

        score = 0.0

        score += 3.0 * len(
            query_words
            & logical_words
        )

        score += 1.0 * len(
            query_words
            & title_words
        )

        normalized_id = self._normalize(
            logical_id
        )

        normalized_query = self._normalize(
            query
        )

        if (
            normalized_id
            and normalized_id
            in normalized_query
        ):
            score += 5.0

        return score

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(
        self,
        query,
        top_k=5,
    ):
        scored = []

        for logical_id, versions in (
            self.lineages.items()
        ):
            score = self._score_lineage(
                logical_id,
                versions,
                query,
            )

            if score <= 0:
                continue

            scored.append(
                (
                    score,
                    logical_id,
                    versions,
                )
            )

        scored.sort(
            key=lambda x: (
                -x[0],
                x[1],
            )
        )

        results = []

        for (
            score,
            logical_id,
            versions,
        ) in scored[:top_k]:

            timeline = []

            for version in versions:
                timeline.append(
                    {
                        "version":
                            version[
                                "version"
                            ],

                        "change_type":
                            version[
                                "change_type"
                            ],

                        "previous_similarity":
                            version[
                                "previous_similarity"
                            ],

                        "fingerprint":
                            version[
                                "fingerprint"
                            ][:12],

                        "code":
                            version[
                                "code"
                            ],
                    }
                )

            results.append(
                {
                    "logical_id":
                        logical_id,

                    "score":
                        score,

                    "version_count":
                        len(versions),

                    "timeline":
                        timeline,
                }
            )

        return {
            "query":
                query,

            "strategy":
                "version_lineage_search",

            "results":
                results,
        }
