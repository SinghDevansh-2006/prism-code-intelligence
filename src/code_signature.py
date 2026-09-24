import ast
import re
import warnings
from collections import Counter


def split_identifier(name):
    name = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        name
    )

    name = name.replace("_", " ")

    return " ".join(
        part.lower()
        for part in name.split()
        if part
    )


def unique_limit(items, limit=30):
    seen = set()
    result = []

    for item in items:
        if not item:
            continue

        item = split_identifier(item)

        if not item:
            continue

        if item not in seen:
            seen.add(item)
            result.append(item)

        if len(result) >= limit:
            break

    return result


class SignatureVisitor(ast.NodeVisitor):

    def __init__(self):
        self.functions = []
        self.classes = []
        self.parameters = []
        self.imports = []
        self.calls = []
        self.attributes = []
        self.names = []

        self.structures = Counter()
        self.operators = Counter()

        self.function_stack = []
        self.recursive_functions = set()

    def _enter_function(self, node):
        self.functions.append(node.name)

        for arg in node.args.args:
            self.parameters.append(arg.arg)

        self.function_stack.append(node.name)

    def _leave_function(self):
        self.function_stack.pop()

    def visit_FunctionDef(self, node):
        self.structures["function"] += 1

        self._enter_function(node)
        self.generic_visit(node)
        self._leave_function()

    def visit_AsyncFunctionDef(self, node):
        self.structures["async function"] += 1

        self._enter_function(node)
        self.generic_visit(node)
        self._leave_function()

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.structures["class"] += 1

        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)

        for alias in node.names:
            self.imports.append(alias.name)

        self.generic_visit(node)

    def visit_Call(self, node):
        call_name = None

        if isinstance(node.func, ast.Name):
            call_name = node.func.id

        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr

        if call_name:
            self.calls.append(call_name)

            if (
                self.function_stack
                and call_name == self.function_stack[-1]
            ):
                self.recursive_functions.add(
                    call_name
                )

        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.attributes.append(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node):
        self.names.append(node.id)
        self.generic_visit(node)

    def visit_For(self, node):
        self.structures["for loop"] += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.structures["async for loop"] += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.structures["while loop"] += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.structures["conditional"] += 1
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.structures["list comprehension"] += 1
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.structures["set comprehension"] += 1
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.structures["dictionary comprehension"] += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.structures["generator expression"] += 1
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.structures["lambda"] += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.structures["exception handling"] += 1
        self.generic_visit(node)

    def visit_With(self, node):
        self.structures["context manager"] += 1
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self.operators[
            type(node.op).__name__
        ] += 1

        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            self.operators[
                type(op).__name__
            ] += 1

        self.generic_visit(node)


def extract_code_signature(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)

    except (SyntaxError, ValueError, MemoryError):
        return "unparsed python code"

    visitor = SignatureVisitor()
    visitor.visit(tree)

    parts = []

    classes = unique_limit(
        visitor.classes,
        10
    )

    functions = unique_limit(
        visitor.functions,
        15
    )

    parameters = unique_limit(
        visitor.parameters,
        20
    )

    imports = unique_limit(
        visitor.imports,
        15
    )

    calls = unique_limit(
        visitor.calls,
        30
    )

    attributes = unique_limit(
        visitor.attributes,
        20
    )

    identifiers = unique_limit(
        visitor.names,
        30
    )

    if classes:
        parts.append(
            "classes: " + ", ".join(classes)
        )

    if functions:
        parts.append(
            "functions: " + ", ".join(functions)
        )

    if parameters:
        parts.append(
            "parameters: " + ", ".join(parameters)
        )

    if imports:
        parts.append(
            "imports: " + ", ".join(imports)
        )

    if calls:
        parts.append(
            "calls: " + ", ".join(calls)
        )

    if attributes:
        parts.append(
            "attributes: " + ", ".join(attributes)
        )

    if identifiers:
        parts.append(
            "identifiers: " + ", ".join(identifiers)
        )

    structures = [
        f"{name} x{count}"
        for name, count
        in visitor.structures.items()
    ]

    if structures:
        parts.append(
            "structure: "
            + ", ".join(structures)
        )

    operators = [
        f"{split_identifier(name)} x{count}"
        for name, count
        in visitor.operators.items()
    ]

    if operators:
        parts.append(
            "operators: "
            + ", ".join(operators)
        )

    recursive = unique_limit(
        visitor.recursive_functions,
        10
    )

    if recursive:
        parts.append(
            "recursive functions: "
            + ", ".join(recursive)
        )

    if not parts:
        return "simple python code"

    return " | ".join(parts)
