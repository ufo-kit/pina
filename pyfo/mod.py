from pycparser import c_ast


def replace(expr, needle, replacement):
    """Replaces a *name* ID in *expr* by *node*"""
    def check_and_replace(node):
        return replacement if node == needle else node

    class Visitor(c_ast.NodeVisitor):
        def visit_Assignment(self, node):
            node.lvalue = check_and_replace(node.lvalue)
            node.rvalue = check_and_replace(node.rvalue)
            replace(node.lvalue, needle, replacement)
            replace(node.rvalue, needle, replacement)

        def visit_UnaryOp(self, node):
            node.expr = check_and_replace(node.expr)
            replace(node.expr, needle, replacement)

        def visit_BinaryOp(self, node):
            node.left = check_and_replace(node.left)
            node.right = check_and_replace(node.right)
            replace(node.left, needle, replacement)
            replace(node.right, needle, replacement)

        def visit_Return(self, node):
            node.expr = check_and_replace(node.expr)
            replace(node.expr, needle, replacement)

        def visit_ExprList(self, node):
            for i, expr in enumerate(node.exprs):
                node.exprs[i] = check_and_replace(node.exprs[i])
                replace(node.exprs[i], needle, replacement)

        def visit_FuncCall(self, node):
            for i, expr, in enumerate(node.args.exprs):
                node.args.exprs[i] = check_and_replace(node.args.exprs[i])
                replace(node.args.exprs[i], needle, replacement)

    Visitor().visit(expr)


def find_read_only(body, params):
    names = [p.name for p in params]

    class Visitor(c_ast.NodeVisitor):
        def __init__(self):
            self.found = None

        def visit_Assignment(self, node):
            for _, c in node.lvalue.children():
                self.found = None
                self.visit(c)

                if self.found:
                    names.remove(self.found)

        def visit_ID(self, node):
            if node.name in names:
                self.found = node.name

    Visitor().visit(body)
    return [p for p in params if p.name in names]


def find_type(c_node, node_type):
    """Find all occurrences of *node_type* in the AST"""
    result = []

    class Visitor(c_ast.NodeVisitor):
        def generic_visit(self, node):
            if isinstance(node, node_type):
                result.append(node)

            for _, c in node.children():
                self.visit(c)

    Visitor().visit(c_node)
    return result


def find_name(body, name):
    result = []

    class Visitor(c_ast.NodeVisitor):
        def __init__(self):
            self.found = None

        def visit_ID(self, node):
            if node.name == name:
                self.found = node

        def generic_visit(self, node):
            for _, c in node.children():
                self.visit(c)

                if self.found:
                    result.append((node, self.found))
                    self.found = None

    for stmt in body.block_items:
        Visitor().visit(stmt)

    return result


def find_statement(c_node, node_type):
    """
    Find all statements in *node* of *node_type* and return a list with tuples
    containing (compound, statement, index).
    """
    result = []

    class Visitor(c_ast.NodeVisitor):
        def visit_Compound(self, node):
            for i, stmt in enumerate(node.block_items):
                if isinstance(stmt, node_type):
                    result.append((node, stmt, i))

            for _, c in node.children():
                self.visit(c)

    Visitor().visit(c_node)
    return result
