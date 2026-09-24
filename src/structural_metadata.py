import ast
import re
import warnings
from collections import defaultdict


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)

        if prefix:
            return f"{prefix}.{node.attr}"

        return node.attr

    return None


class StructuralVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.calls = []
        self.identifiers = set()
        self.loops = []

        self.recursive_functions = set()

        self.function_stack = []
        self.class_stack = []

        self.loop_depth = 0

    # --------------------------------------------------
    # CURRENT SCOPE
    # --------------------------------------------------

    @property
    def current_function(self):
        if not self.function_stack:
            return None

        return self.function_stack[-1]

    @property
    def current_class(self):
        if not self.class_stack:
            return None

        return self.class_stack[-1]

    # --------------------------------------------------
    # FUNCTIONS
    # --------------------------------------------------

    def visit_FunctionDef(self, node):
        self.functions.append(
            {
                "name": node.name,
                "line": getattr(
                    node,
                    "lineno",
                    None
                ),
                "class":
                    self.current_class,
            }
        )

        self.function_stack.append(
            node.name
        )

        self.generic_visit(node)

        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self.functions.append(
            {
                "name": node.name,
                "line": getattr(
                    node,
                    "lineno",
                    None
                ),
                "class":
                    self.current_class,
            }
        )

        self.function_stack.append(
            node.name
        )

        self.generic_visit(node)

        self.function_stack.pop()

    # --------------------------------------------------
    # CLASSES
    # --------------------------------------------------

    def visit_ClassDef(self, node):
        self.classes.append(
            {
                "name": node.name,
                "line": getattr(
                    node,
                    "lineno",
                    None
                ),
            }
        )

        self.class_stack.append(
            node.name
        )

        self.generic_visit(node)

        self.class_stack.pop()

    # --------------------------------------------------
    # IMPORTS
    # --------------------------------------------------

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(
                {
                    "name": alias.name,
                    "line": getattr(
                        node,
                        "lineno",
                        None
                    ),
                }
            )

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""

        for alias in node.names:
            if module:
                name = (
                    f"{module}."
                    f"{alias.name}"
                )
            else:
                name = alias.name

            self.imports.append(
                {
                    "name": name,
                    "line": getattr(
                        node,
                        "lineno",
                        None
                    ),
                }
            )

        self.generic_visit(node)

    # --------------------------------------------------
    # CALLS
    # --------------------------------------------------

    def visit_Call(self, node):
        name = dotted_name(
            node.func
        )

        if name:
            self.calls.append(
                {
                    "name": name,
                    "line": getattr(
                        node,
                        "lineno",
                        None
                    ),
                    "function":
                        self.current_function,
                    "class":
                        self.current_class,
                }
            )

            final_component = (
                name.split(".")[-1]
            )

            if (
                self.current_function
                and final_component
                == self.current_function
            ):
                self.recursive_functions.add(
                    self.current_function
                )

        self.generic_visit(node)

    # --------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------

    def visit_Name(self, node):
        self.identifiers.add(
            node.id
        )

        self.generic_visit(node)

    def visit_Attribute(self, node):
        name = dotted_name(node)

        if name:
            self.identifiers.add(
                name
            )

        self.generic_visit(node)

    # --------------------------------------------------
    # LOOPS
    # --------------------------------------------------

    def _visit_loop(self, node, loop_type):
        self.loop_depth += 1

        self.loops.append(
            {
                "type": loop_type,
                "line": getattr(
                    node,
                    "lineno",
                    None
                ),
                "depth":
                    self.loop_depth,
                "function":
                    self.current_function,
                "class":
                    self.current_class,
            }
        )

        self.generic_visit(node)

        self.loop_depth -= 1

    def visit_For(self, node):
        self._visit_loop(
            node,
            "for"
        )

    def visit_AsyncFor(self, node):
        self._visit_loop(
            node,
            "async_for"
        )

    def visit_While(self, node):
        self._visit_loop(
            node,
            "while"
        )


def extract_structural_metadata(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )

            tree = ast.parse(code)

    except (
        SyntaxError,
        ValueError,
        MemoryError
    ):
        identifiers = sorted(
            set(
                re.findall(
                    r"\b[A-Za-z_]"
                    r"[A-Za-z0-9_.]*\b",
                    code
                )
            )
        )

        return {
            "parsed": False,
            "functions": [],
            "classes": [],
            "imports": [],
            "calls": [],
            "identifiers":
                identifiers[:300],
            "recursive_functions": [],
            "loops": [],
            "loop_count": 0,
            "max_loop_depth": 0,
            "class_loop_depths": {},
            "function_loop_depths": {},
        }

    visitor = StructuralVisitor()
    visitor.visit(tree)

    class_depths = defaultdict(int)
    function_depths = defaultdict(int)

    for loop in visitor.loops:
        depth = loop["depth"]

        if loop["class"]:
            class_depths[
                loop["class"]
            ] = max(
                class_depths[
                    loop["class"]
                ],
                depth
            )

        if loop["function"]:
            function_depths[
                loop["function"]
            ] = max(
                function_depths[
                    loop["function"]
                ],
                depth
            )

    max_depth = max(
        [
            loop["depth"]
            for loop in visitor.loops
        ],
        default=0
    )

    return {
        "parsed": True,

        "functions":
            visitor.functions,

        "classes":
            visitor.classes,

        "imports":
            visitor.imports,

        "calls":
            visitor.calls,

        "identifiers":
            sorted(
                visitor.identifiers
            )[:300],

        "recursive_functions":
            sorted(
                visitor.recursive_functions
            ),

        "loops":
            visitor.loops,

        "loop_count":
            len(visitor.loops),

        "max_loop_depth":
            max_depth,

        "class_loop_depths":
            dict(class_depths),

        "function_loop_depths":
            dict(function_depths),
    }
