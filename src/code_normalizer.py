import ast


class LargeLiteralCompactor(ast.NodeTransformer):
    def visit_List(self, node):
        self.generic_visit(node)

        if len(node.elts) > 50:
            preview = node.elts[:8]

            marker = ast.Constant(
                value=f"<LONG_LIST:{len(node.elts)}_ITEMS>"
            )

            return ast.copy_location(
                ast.List(
                    elts=preview + [marker],
                    ctx=ast.Load()
                ),
                node
            )

        return node

    def visit_Tuple(self, node):
        self.generic_visit(node)

        if len(node.elts) > 50:
            preview = node.elts[:8]

            marker = ast.Constant(
                value=f"<LONG_TUPLE:{len(node.elts)}_ITEMS>"
            )

            return ast.copy_location(
                ast.Tuple(
                    elts=preview + [marker],
                    ctx=ast.Load()
                ),
                node
            )

        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) > 1000:
            preview = node.value[:120]

            return ast.copy_location(
                ast.Constant(
                    value=(
                        preview
                        + f"...<LONG_STRING:{len(node.value)}_CHARS>"
                    )
                ),
                node
            )

        return node


def compact_large_literals(code: str) -> str:
    try:
        tree = ast.parse(code)
        tree = LargeLiteralCompactor().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError, MemoryError):
        return code
