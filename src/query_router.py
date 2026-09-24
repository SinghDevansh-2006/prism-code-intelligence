import re
from dataclasses import dataclass


@dataclass
class QueryPlan:
    route: str
    confidence: float
    reason: str
    extracted_terms: list[str]


# --------------------------------------------------
# ROUTE PATTERNS
# --------------------------------------------------

EVOLUTION_PATTERNS = [
    r"\bversion\b",
    r"\bversions\b",
    r"\bhistory\b",
    r"\bchanged\b",
    r"\bchange(s|d)? between\b",
    r"\bprevious implementation\b",
    r"\bolder implementation\b",
    r"\bevolved\b",
    r"\bwhen was\b",
]


# High-specificity structural patterns go BEFORE exact usage.
STRUCTURAL_PATTERNS = [
    r"\b(before|after)\b",
    r"\brecursive\b",
    r"\brecursion\b",
    r"\bnested loops?\b",
    r"\bcontains?\b.*\bloop\b",
    r"\bclass(es)?\b.*\bwith\b",
    r"\bfunctions?\b.*\bwith\b",
    r"\bcall graph\b",
    r"\bdependency graph\b",
    r"\bsequence of calls\b",
]


EXACT_USAGE_PATTERNS = [
    r"\bwhere is\b.*\bused\b",
    r"\bfind usages?\b",
    r"\bfind references?\b",
    r"\bwhich files import\b",
    r"\bwhich functions call\b",
    r"\bwhich methods call\b",
    r"\buses? [`'\"]?[A-Za-z_][A-Za-z0-9_.]*",
    r"\bcall(s|ing)? [`'\"]?[A-Za-z_][A-Za-z0-9_.]*",
    r"\bimport(s|ing)? [`'\"]?[A-Za-z_][A-Za-z0-9_.]*",
]


IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_.]*\b"
)


STOPWORDS = {
    "find",
    "code",
    "that",
    "which",
    "where",
    "what",
    "uses",
    "use",
    "using",
    "usages",
    "usage",
    "used",
    "calls",
    "call",
    "calling",
    "files",
    "file",
    "functions",
    "function",
    "methods",
    "method",
    "classes",
    "class",
    "before",
    "after",
    "with",
    "implementation",
    "implementations",
    "version",
    "versions",
    "history",
    "changed",
    "change",
    "between",
    "older",
    "previous",
    "show",
    "how",
    "this",
    "contains",
    "contain",
    "import",
    "imports",
    "importing",
    "references",
    "reference",
    "recursive",
    "recursion",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "is",
    "are",
}


def _matches_any(query, patterns):
    return any(
        re.search(
            pattern,
            query,
            flags=re.IGNORECASE
        )
        for pattern in patterns
    )


def _extract_terms(query):
    candidates = IDENTIFIER_PATTERN.findall(
        query
    )

    result = []
    seen = set()

    for term in candidates:
        lowered = term.lower()

        if lowered in STOPWORDS:
            continue

        if len(term) <= 1:
            continue

        if term not in seen:
            seen.add(term)
            result.append(term)

    return result[:12]


def route_query(query):
    query = query.strip()

    terms = _extract_terms(
        query
    )

    # --------------------------------------------------
    # 1. EVOLUTION
    # --------------------------------------------------

    if _matches_any(
        query,
        EVOLUTION_PATTERNS
    ):
        return QueryPlan(
            route="evolution",
            confidence=0.95,
            reason=(
                "Query explicitly asks about "
                "versions, history, or code changes."
            ),
            extracted_terms=terms,
        )

    # --------------------------------------------------
    # 2. STRUCTURAL
    #
    # Must come before exact usage because:
    #
    # "call A before B"
    #
    # contains exact identifiers, but the constraint
    # is about code structure / ordering.
    # --------------------------------------------------

    if _matches_any(
        query,
        STRUCTURAL_PATTERNS
    ):
        return QueryPlan(
            route="structural",
            confidence=0.90,
            reason=(
                "Query contains code-structure, "
                "control-flow, or call-order constraints."
            ),
            extracted_terms=terms,
        )

    # --------------------------------------------------
    # 3. EXACT USAGE
    # --------------------------------------------------

    if _matches_any(
        query,
        EXACT_USAGE_PATTERNS
    ):
        return QueryPlan(
            route="exact_usage",
            confidence=0.90,
            reason=(
                "Query asks for concrete identifier "
                "usage, references, calls, or imports."
            ),
            extracted_terms=terms,
        )

    # --------------------------------------------------
    # 4. DEFAULT SEMANTIC RETRIEVAL
    # --------------------------------------------------

    return QueryPlan(
        route="semantic",
        confidence=0.90,
        reason=(
            "Query primarily describes desired "
            "program behavior or functionality."
        ),
        extracted_terms=terms,
    )
