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
            for i, expr in enumerate(node.args.exprs):
                node.args.exprs[i] = check_and_replace(node.args.exprs[i])
                replace(node.args.exprs[i], needle, replacement)

        def visit_Compound(self, node):
            for i, item in enumerate(node.block_items):
                node.block_items[i] = check_and_replace(item)
                replace(node.block_items[i], needle, replacement)

        def visit_ParamList(self, node):
            for i, item in enumerate(node.params):
                node.params[i] = check_and_replace(item)
                replace(node.params[i], needle, replacement)


    Visitor().visit(expr)


def find(node, cond):
    """
    Find nodes in *node* that satisfy *cond*, a callable receiving a single
    node.
    """
    result = []

    class Visitor(c_ast.NodeVisitor):
        def generic_visit(self, node):
            if cond(node):
                result.append(node)

            for _, c in node.children():
                result.extend(find(c, cond))

    Visitor().visit(node)
    return result


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
    return find(c_node, lambda node: isinstance(node, node_type))


def find_name(body, name):
    """Find all identifier nodes with *name*"""
    return find(body, lambda node: isinstance(node, c_ast.ID) and node.name == name)


def find_global_names(fdef):
    """Return all parameter names of *fdef*"""
    return (p.name for p in fdef.decl.type.args.params)


def TypeDecl(name, typename, init):
    """Create a simple type declaration such as '*typename* *name* = *init*'"""
    idtype = c_ast.IdentifierType([typename])
    typedecl = c_ast.TypeDecl(name, [], idtype)
    return c_ast.Decl(name, None, None, None, typedecl, init, None)


def PtrDecl(name, typename, qualifiers):
    """Create a pointer type declaration such as '*typename* * *name*'"""
    idtype = c_ast.IdentifierType([typename])
    typedecl = c_ast.TypeDecl(name, None, idtype)
    ptrdecl = c_ast.PtrDecl(None, typedecl)
    return c_ast.Decl(name, qualifiers, None, None, ptrdecl, None, None)


def ArrayRef(name, subscript):
    """Create an array reference such as '*name*[*subscript*]'"""
    return c_ast.ArrayRef(c_ast.ID(name), c_ast.ID(subscript))


def WorkItemIndex(dims=2):
    """Build an expression to build the current work item index."""
    left = c_ast.ID('get_global_id(1)')
    right = c_ast.ID('get_global_size(0)')
    a = c_ast.BinaryOp('*', left, right)
    return c_ast.BinaryOp('+', a, c_ast.ID('get_global_id(0)'))
