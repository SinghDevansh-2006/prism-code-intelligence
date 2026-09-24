import hashlib
import json
import time
from pathlib import Path

from src.version_index import (
    normalize_code,
    token_set,
    jaccard_similarity,
)


class IncrementalVersionStore:
    def __init__(
        self,
        path="runtime_index/version_store.json",
        similarity_threshold=0.82,
    ):
        self.path = Path(path)

        self.similarity_threshold = (
            similarity_threshold
        )

        self.lineages = {}

        if self.path.exists():
            self._load()

    # --------------------------------------------------
    # LOAD / SAVE
    # --------------------------------------------------

    def _load(self):
        with open(self.path) as f:
            payload = json.load(f)

        self.similarity_threshold = (
            payload[
                "similarity_threshold"
            ]
        )

        self.lineages = payload[
            "lineages"
        ]

    def save(self):
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        payload = {
            "similarity_threshold":
                self.similarity_threshold,

            "lineages":
                self.lineages,
        }

        with open(
            self.path,
            "w"
        ) as f:
            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False,
            )

    # --------------------------------------------------
    # FINGERPRINT
    # --------------------------------------------------

    @staticmethod
    def fingerprint(code):
        normalized = normalize_code(
            code
        )

        return hashlib.sha256(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()

    # --------------------------------------------------
    # ADD ONE VERSION
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
        start = time.perf_counter()

        fingerprint = self.fingerprint(
            code
        )

        lineage = self.lineages.setdefault(
            logical_id,
            []
        )

        # Prevent duplicate version IDs.
        for item in lineage:
            if (
                str(item["version"])
                == str(version)
            ):
                raise ValueError(
                    f"Version {version} already "
                    f"exists for {logical_id}"
                )

        # --------------------------------------------------
        # COMPARE ONLY AGAINST LATEST VERSION
        # --------------------------------------------------

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
                previous_tokens = token_set(
                    previous["code"]
                )

                current_tokens = token_set(
                    code
                )

                similarity = (
                    jaccard_similarity(
                        previous_tokens,
                        current_tokens,
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

        record = {
            "logical_id":
                logical_id,

            "version":
                version,

            "title":
                title,

            "language":
                language,

            "code":
                code,

            "metadata":
                metadata or {},

            "fingerprint":
                fingerprint,

            "previous_similarity":
                similarity,

            "change_type":
                change_type,
        }

        lineage.append(
            record
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return {
            "record":
                record,

            "index_update_ms":
                elapsed_ms,
        }

    # --------------------------------------------------
    # HISTORY
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
    # STATS
    # --------------------------------------------------

    def stats(self):
        version_count = sum(
            len(items)
            for items
            in self.lineages.values()
        )

        unchanged = 0
        near_duplicate = 0
        substantial = 0

        for items in self.lineages.values():
            for item in items:
                change_type = item[
                    "change_type"
                ]

                if (
                    change_type
                    == "unchanged"
                ):
                    unchanged += 1

                elif (
                    change_type
                    == "near_duplicate_update"
                ):
                    near_duplicate += 1

                elif (
                    change_type
                    == "substantial_update"
                ):
                    substantial += 1

        return {
            "logical_snippets":
                len(self.lineages),

            "versions":
                version_count,

            "unchanged_versions":
                unchanged,

            "near_duplicate_updates":
                near_duplicate,

            "substantial_updates":
                substantial,
        }
