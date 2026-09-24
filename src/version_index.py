import ast
import hashlib
import json
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path


def normalize_code(code):
    """
    Produce a formatting-insensitive representation when possible.
    Falls back to whitespace normalization for unparsable code.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)

        return ast.dump(
            tree,
            annotate_fields=True,
            include_attributes=False,
        )

    except (SyntaxError, ValueError, MemoryError):
        return re.sub(
            r"\s+",
            " ",
            code
        ).strip()


def exact_fingerprint(code):
    normalized = normalize_code(code)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def token_set(code):
    return set(
        re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*"
            r"|\d+"
            r"|==|!=|<=|>=|//|\*\*"
            r"|[+\-*/%<>]",
            code.lower(),
        )
    )


def jaccard_similarity(a, b):
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


class VersionIndex:
    def __init__(
        self,
        similarity_threshold=0.82,
    ):
        self.similarity_threshold = (
            similarity_threshold
        )

        self.records = []
        self.lineages = defaultdict(list)

    def add_snapshot(
        self,
        logical_id,
        version,
        code,
        title="",
        language="python",
        metadata=None,
    ):
        record = {
            "logical_id": logical_id,
            "version": version,
            "title": title,
            "language": language,
            "code": code,
            "metadata": metadata or {},
            "fingerprint":
                exact_fingerprint(code),
        }

        self.records.append(record)

        return record

    def build_lineages(self):
        """
        Group versions belonging to the same logical snippet.

        Within each logical ID, annotate how similar each version is
        to the previous version.
        """
        grouped = defaultdict(list)

        for record in self.records:
            grouped[
                record["logical_id"]
            ].append(record)

        self.lineages = defaultdict(list)

        for logical_id, versions in grouped.items():
            # Preserve caller-provided version ordering by sorting
            # naturally as strings/numbers where possible.
            versions = sorted(
                versions,
                key=lambda x: str(x["version"])
            )

            previous_tokens = None
            previous_fp = None

            for position, record in enumerate(
                versions
            ):
                current_tokens = token_set(
                    record["code"]
                )

                if position == 0:
                    similarity = None
                    change_type = "initial"

                elif (
                    record["fingerprint"]
                    == previous_fp
                ):
                    similarity = 1.0
                    change_type = "unchanged"

                else:
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

                enriched = {
                    **record,
                    "previous_similarity":
                        similarity,
                    "change_type":
                        change_type,
                }

                self.lineages[
                    logical_id
                ].append(enriched)

                previous_tokens = current_tokens
                previous_fp = record[
                    "fingerprint"
                ]

        return self.lineages

    def history(self, logical_id):
        if not self.lineages:
            self.build_lineages()

        return self.lineages.get(
            logical_id,
            []
        )

    def stats(self):
        if not self.lineages:
            self.build_lineages()

        versions = sum(
            len(items)
            for items in self.lineages.values()
        )

        unchanged = 0
        near_duplicate = 0
        substantial = 0

        for items in self.lineages.values():
            for item in items:
                if (
                    item["change_type"]
                    == "unchanged"
                ):
                    unchanged += 1

                elif (
                    item["change_type"]
                    == "near_duplicate_update"
                ):
                    near_duplicate += 1

                elif (
                    item["change_type"]
                    == "substantial_update"
                ):
                    substantial += 1

        return {
            "logical_snippets":
                len(self.lineages),
            "versions": versions,
            "unchanged_versions":
                unchanged,
            "near_duplicate_updates":
                near_duplicate,
            "substantial_updates":
                substantial,
        }

    def save(self, path):
        if not self.lineages:
            self.build_lineages()

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        payload = {
            "similarity_threshold":
                self.similarity_threshold,
            "stats":
                self.stats(),
            "lineages":
                dict(self.lineages),
        }

        with open(
            path,
            "w"
        ) as f:
            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return path
