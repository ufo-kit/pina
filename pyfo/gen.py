import parser
import qualifiers
from pycparser import c_generator, c_ast


def work_item_index(dims=2):
    """Build an expression to build the current work item index."""
    left = c_ast.ID('get_global_id(1)')
    right = c_ast.ID('get_global_size(0)')
    a = c_ast.BinaryOp('*', left, right)
    return c_ast.BinaryOp('+', a, c_ast.ID('get_global_id(0)'))


def array_ref(name, subscript):
    return c_ast.ArrayRef(c_ast.ID(name), c_ast.ID(subscript))


def ptr_decl(name, vtype, qualifiers):
    """Create a pointer declaration"""
    idtype = c_ast.IdentifierType([vtype])
    typedecl = c_ast.TypeDecl(name, [], idtype)
    ptrdecl = c_ast.PtrDecl([], typedecl)
    return c_ast.Decl(name, None, None, None, qualifiers, ptrdecl, None, None)


def local_decl(name, vtype, init=None):
    typedecl =  c_ast.TypeDecl(name, [], c_ast.IdentifierType([vtype]))
    return c_ast.Decl(name, None, None, None, None, typedecl, init, None)


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


def replace(expr, needle, replacement):
    """Replaces a *name* ID in *expr* by *node*"""
    def check_and_replace(node):
        return replacement if node == needle else node

    class Visitor(c_ast.NodeVisitor):
        def visit_Assignment(self, node):
            node.lvalue = check_and_replace(node.lvalue)
            node.rvalue = check_and_replace(node.rvalue)

        def visit_BinaryOp(self, node):
            node.left = check_and_replace(node.left)
            node.right = check_and_replace(node.right)

    Visitor().visit(expr)


def kernel(func, arg_types=[]):
    """Build OpenCL kernel source string from *func*"""
    fdef = parser.parse(func)
    params = fdef.decl.type.args.params

    # add __kernel qualifier to signature
    fdef.decl.type.type.quals = ['__kernel']

    # assign arg types
    mapped = list(enumerate(zip(params, arg_types)))

    for i, (node, qualifier) in mapped:
        params[i] = ptr_decl(node.name, 'float', [qualifier.cl_keyword])

    # create work item indices
    fdef.body.block_items.insert(0, local_decl('idx', 'int', work_item_index()))

    # create variables for intermediate results
    localvars = []
    assignments = find_type(fdef.body, c_ast.Assignment)

    for each in assignments:
        if not each.lvalue.name in localvars:
            localvars.append(each.lvalue.name)

    for var in localvars:
        fdef.body.block_items.insert(0, local_decl(var, 'float'))

    # replace global occurences with array access
    globalvars = [p.name for p in params]

    for name in globalvars:
        for expr, node in find_name(fdef.body, name):
            read_access = array_ref(name, 'idx')
            replace(expr, node, read_access)

    # replace return with memory write
    lvalue = array_ref('out', 'idx')

    for compound, stmt, i in find_statement(fdef, c_ast.Return):
        compound.block_items[i] = c_ast.Assignment('=', lvalue, stmt.expr)

    # add out argument
    fdef.decl.type.args.params.append(ptr_decl('out', 'float', ['__global']))

    generator = c_generator.CGenerator()
    return generator.visit(fdef)
