import json
import re
from pathlib import Path

from src.query_router import route_query


class StructuralSearchEngine:
    def __init__(
        self,
        index_dir="runtime_index",
    ):
        index_dir = Path(index_dir)

        self.records = []

        with open(
            index_dir / "structural_index.jsonl"
        ) as f:
            for line in f:
                self.records.append(
                    json.loads(line)
                )

        self.corpus = {}

        with open(
            index_dir / "corpus.jsonl"
        ) as f:
            for line in f:
                row = json.loads(line)

                self.corpus[
                    row["id"]
                ] = row

    # --------------------------------------------------
    # MATCH HELPERS
    # --------------------------------------------------

    @staticmethod
    def _matches(name, term):
        if not name:
            return False

        return (
            name == term
            or name.endswith("." + term)
        )

    def _matching_lines(
        self,
        record,
        term,
        categories=None,
    ):
        matches = []

        for category in (categories or ["functions", "classes", "imports", "calls"]):
            for item in record[category]:
                if self._matches(
                    item["name"],
                    term
                ) or (category == "imports" and item["name"].startswith(term + ".")):
                    evidence = {
                        "type":
                            category[:-1],
                        "name":
                            item["name"],
                        "line":
                            item.get("line"),
                    }

                    if "function" in item:
                        evidence[
                            "function"
                        ] = item.get(
                            "function"
                        )

                    if "class" in item:
                        evidence[
                            "class"
                        ] = item.get(
                            "class"
                        )

                    matches.append(
                        evidence
                    )

        return matches

    # --------------------------------------------------
    # EXACT USAGE
    # --------------------------------------------------

    def exact_usage(
        self,
        terms,
        top_k=20,
        categories=None,
        require_function=False,
    ):
        results = []

        for record in self.records:
            evidence = []
            matched_terms = 0

            for term in terms:
                matches = (
                    self._matching_lines(
                        record,
                        term, categories=categories
                    )
                )

                if require_function:
                    matches = [m for m in matches if m.get("function") is not None]

                if matches:
                    matched_terms += 1
                    evidence.extend(
                        matches
                    )

                elif categories is None:
                    identifier_match = any(
                        self._matches(
                            identifier,
                            term
                        )
                        for identifier
                        in record[
                            "identifiers"
                        ]
                    )

                    if identifier_match:
                        matched_terms += 1

                        evidence.append(
                            {
                                "type":
                                    "identifier",
                                "name": term,
                                "line": None,
                            }
                        )

            if matched_terms == 0 or (categories is not None and matched_terms != len(terms)):
                continue

            score = (
                matched_terms
                / max(
                    len(terms),
                    1
                )
            )

            results.append(
                {
                    "id": record["id"],
                    "score": score,
                    "evidence":
                        evidence[:12],
                }
            )

        results.sort(
            key=lambda x: (
                -x["score"],
                x["id"]
            )
        )

        return results[:top_k]

    # --------------------------------------------------
    # SAME-SCOPE CALL ORDER
    # --------------------------------------------------

    def call_order(
        self,
        first_term,
        second_term,
        top_k=20,
        require_function=False,
    ):
        results = []

        for record in self.records:
            first_calls = [
                call
                for call in record["calls"]
                if self._matches(
                    call["name"],
                    first_term
                )
            ]

            second_calls = [
                call
                for call in record["calls"]
                if self._matches(
                    call["name"],
                    second_term
                )
            ]

            if (
                not first_calls
                or not second_calls
            ):
                continue

            valid_pairs = []

            for first in first_calls:
                for second in second_calls:
                    first_function = (
                        first.get(
                            "function"
                        )
                    )

                    second_function = (
                        second.get(
                            "function"
                        )
                    )

                    first_class = (
                        first.get(
                            "class"
                        )
                    )

                    second_class = (
                        second.get(
                            "class"
                        )
                    )

                    same_function = (
                        first_function
                        == second_function
                    )

                    same_class = (
                        first_class
                        == second_class
                    )

                    if not same_function:
                        continue

                    if not same_class:
                        continue

                    if (
                        require_function
                        and first_function is None
                    ):
                        continue

                    if (
                        first["line"] is None
                        or second["line"] is None
                    ):
                        continue

                    if (
                        first["line"]
                        >= second["line"]
                    ):
                        continue

                    valid_pairs.append(
                        (
                            first,
                            second
                        )
                    )

            if not valid_pairs:
                continue

            first, second = (
                valid_pairs[0]
            )

            scope = (
                first.get("function")
                or "<module>"
            )

            results.append(
                {
                    "id":
                        record["id"],

                    "score":
                        1.0,

                    "scope":
                        scope,

                    "class":
                        first.get(
                            "class"
                        ),

                    "evidence": [
                        {
                            "type":
                                "call_order",
                            "name":
                                first["name"],
                            "line":
                                first["line"],
                            "function":
                                first.get(
                                    "function"
                                ),
                            "class":
                                first.get(
                                    "class"
                                ),
                        },
                        {
                            "type":
                                "call_order",
                            "name":
                                second["name"],
                            "line":
                                second["line"],
                            "function":
                                second.get(
                                    "function"
                                ),
                            "class":
                                second.get(
                                    "class"
                                ),
                        },
                    ],
                }
            )

        return results[:top_k]

    # --------------------------------------------------
    # STRUCTURAL SEARCH
    # --------------------------------------------------

    def structural(
        self,
        query,
        terms,
        top_k=20,
    ):
        lowered = query.lower()

        # --------------------------------------------------
        # CALL ORDER
        # --------------------------------------------------

        if (
            (
                " before " in lowered
                or " after " in lowered
            )
            and len(terms) >= 2
        ):
            require_function = (
                "function" in lowered
                or "method" in lowered
            )

            if " before " in lowered:
                return self.call_order(
                    terms[0],
                    terms[1],
                    top_k=top_k,
                    require_function=
                        require_function,
                )

            return self.call_order(
                terms[1],
                terms[0],
                top_k=top_k,
                require_function=
                    require_function,
            )

        if not any(term in lowered for term in ("recursive", "recursion", "nested loop")):
            raise ValueError("Supported structural queries cover recursion, nested loops, or two named calls before/after each other; call/dependency graphs are not implemented.")

        results = []

        for record in self.records:
            evidence = []
            score = 0.0

            # --------------------------------------------------
            # RECURSION
            # --------------------------------------------------

            if (
                "recursive" in lowered
                or "recursion" in lowered
            ):
                recursive = record[
                    "recursive_functions"
                ]

                if not recursive:
                    continue

                score += 2.0

                for name in recursive:
                    function_info = next(
                        (
                            f
                            for f in record[
                                "functions"
                            ]
                            if f["name"]
                            == name
                        ),
                        None
                    )

                    evidence.append(
                        {
                            "type":
                                "recursive_function",
                            "name": name,
                            "line":
                                (
                                    function_info[
                                        "line"
                                    ]
                                    if function_info
                                    else None
                                ),
                            "class":
                                (
                                    function_info.get(
                                        "class"
                                    )
                                    if function_info
                                    else None
                                ),
                        }
                    )

            # --------------------------------------------------
            # CLASS-SCOPED NESTED LOOPS
            # --------------------------------------------------

            if (
                "nested loop" in lowered
                and "class" in lowered
            ):
                class_matches = [
                    (
                        class_name,
                        depth
                    )
                    for (
                        class_name,
                        depth
                    )
                    in record[
                        "class_loop_depths"
                    ].items()
                    if depth >= 2
                ]

                if not class_matches:
                    continue

                best_depth = max(
                    depth
                    for _, depth
                    in class_matches
                )

                score += best_depth

                for (
                    class_name,
                    depth
                ) in class_matches:
                    evidence.append(
                        {
                            "type":
                                "nested_loops_in_class",
                            "name":
                                class_name,
                            "depth":
                                depth,
                            "line": None,
                        }
                    )

            # --------------------------------------------------
            # FUNCTION-SCOPED NESTED LOOPS
            # --------------------------------------------------

            elif (
                "nested loop" in lowered
                and (
                    "function" in lowered
                    or "method" in lowered
                )
            ):
                function_matches = [
                    (
                        function_name,
                        depth
                    )
                    for (
                        function_name,
                        depth
                    )
                    in record[
                        "function_loop_depths"
                    ].items()
                    if depth >= 2
                ]

                if not function_matches:
                    continue

                best_depth = max(
                    depth
                    for _, depth
                    in function_matches
                )

                score += best_depth

                for (
                    function_name,
                    depth
                ) in function_matches:
                    evidence.append(
                        {
                            "type":
                                "nested_loops_in_function",
                            "name":
                                function_name,
                            "depth":
                                depth,
                            "line": None,
                        }
                    )

            # --------------------------------------------------
            # GENERIC NESTED LOOPS
            # --------------------------------------------------

            elif "nested loop" in lowered:
                if (
                    record[
                        "max_loop_depth"
                    ] < 2
                ):
                    continue

                score += record[
                    "max_loop_depth"
                ]

                evidence.append(
                    {
                        "type":
                            "loop_structure",
                        "name":
                            "nested loops",
                        "depth":
                            record[
                                "max_loop_depth"
                            ],
                        "line":
                            None,
                    }
                )

            # --------------------------------------------------
            # IDENTIFIER EVIDENCE
            # --------------------------------------------------

            term_matches = 0

            for term in terms:
                matches = (
                    self._matching_lines(
                        record,
                        term
                    )
                )

                if matches:
                    term_matches += 1
                    evidence.extend(
                        matches[:4]
                    )

            score += (
                0.25
                * term_matches
            )

            if score <= 0:
                continue

            results.append(
                {
                    "id":
                        record["id"],
                    "score":
                        score,
                    "evidence":
                        evidence[:12],
                }
            )

        results.sort(
            key=lambda x: (
                -x["score"],
                x["id"]
            )
        )

        return results[:top_k]

    # --------------------------------------------------
    # ROUTED SEARCH
    # --------------------------------------------------

    def search(
        self,
        query,
        top_k=10,
    ):
        plan = route_query(
            query
        )

        if (
            plan.route
            == "exact_usage"
        ):
            categories = None
            if re.search(r"\bimports?\b", query, re.I):
                categories = ["imports"]
            elif re.search(r"\bcalls?\b", query, re.I):
                categories = ["calls"]
            matches = self.exact_usage(
                plan.extracted_terms, top_k=top_k, categories=categories,
                require_function=categories == ["calls"] and bool(re.search(r"\b(functions?|methods?)\b", query, re.I)),
            )

        elif (
            plan.route
            == "structural"
        ):
            matches = self.structural(
                query,
                plan.extracted_terms,
                top_k=top_k,
            )

        else:
            raise ValueError(
                "Unsupported symbolic route: "
                f"{plan.route}"
            )

        results = []

        for match in matches:
            document = self.corpus[
                match["id"]
            ]

            results.append(
                {
                    **match,
                    "code":
                        document["code"],
                    "title":
                        document["title"],
                    "language":
                        document["language"],
                }
            )

        return {
            "query":
                query,

            "route":
                plan.route,

            "reason":
                plan.reason,

            "terms":
                plan.extracted_terms,

            "results":
                results,
        }
