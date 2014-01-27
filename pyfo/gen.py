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


def decl(name, typename, init):
    idtype = c_ast.IdentifierType([typename])
    typedecl = c_ast.TypeDecl(name, [], idtype)
    return c_ast.Decl(name, None, None, None, None, typedecl, init, None)


def ptr_decl(name, typename, qualifiers):
    """Create a pointer declaration"""
    idtype = c_ast.IdentifierType([typename])
    typedecl = c_ast.TypeDecl(name, [], idtype)
    ptrdecl = c_ast.PtrDecl([], typedecl)
    return c_ast.Decl(name, None, None, None, qualifiers, ptrdecl, None, None)


def local_decl(name, typename, init=None):
    typedecl =  c_ast.TypeDecl(name, [], c_ast.IdentifierType([typename]))
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

        def visit_UnaryOp(self, node):
            node.expr = check_and_replace(node.expr)

        def visit_BinaryOp(self, node):
            node.left = check_and_replace(node.left)
            node.right = check_and_replace(node.right)

        def visit_Return(self, node):
            node.expr = check_and_replace(node.expr)

        def visit_ExprList(self, node):
            for i, expr in enumerate(node.exprs):
                node.exprs[i] = check_and_replace(node.exprs[i])

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


def constantify(fdef, specs, env):
    """Replace small read-only with constant memory"""
    params = fdef.decl.type.args.params
    readonly_params = find_read_only(fdef.body, params)

    constant_size = env.MAX_CONSTANT_SIZE
    constant_args = env.MAX_CONSTANT_ARGS

    for p in readonly_params:
        if constant_args == 0:
            break

        if p.name in specs:
            spec = specs[p.name]

            if spec.size and spec.size < constant_size:
                constant_size -= spec.size
                constant_args -= 1

                p.funcspec = ['__constant']


def substitute_mad(fdef, specs, env):
    """Substitute "a * b + c" expressions  with "mad(a, b, c)"."""
    result = []

    class MulVisitor(c_ast.NodeVisitor):
        def __init__(self):
            self.left = None
            self.right = None

        def visit_BinaryOp(self, node):
            if node.op == '*':
                self.left = node.left
                self.right = node.right

    class AddVisitor(c_ast.NodeVisitor):
        def visit_BinaryOp(self, node):
            if node.op in ('+', '-'):
                v = MulVisitor()
                v.visit(node.left)

                if v.left and v.right:
                    right = node.right

                    if node.op == '-':
                        # invert c to be able to use mad()
                        right = c_ast.UnaryOp('-', right)

                    result.append((node, v.left, v.right, right))

    AddVisitor().visit(fdef.body)

    for node, a, b, c in result:
        # TODO: check that a, b and c are of some float type
        args = c_ast.ExprList([a, b, c])
        mad = c_ast.FuncCall(c_ast.ID('mad'), args)
        replace(fdef, node, mad)


def optimize_depending_on_env(fdef, specs, env):
    substitute_mad(fdef, specs, env)
    constantify(fdef, specs, env)


def fix_for_loops(fdef, specs):
    loops = [l for l in find_type(fdef.body, c_ast.For) if hasattr(l, '_extra')]

    for loop in loops:
        it, mem = loop._extra

        if mem in specs:
            it_var = c_ast.ID(it + '__it')
            n_it = specs[mem].size / 4
            loop.init = decl(it_var.name, 'int', c_ast.Constant('int', '0'))
            loop.cond = c_ast.BinaryOp('<', it_var, c_ast.Constant('int', str(n_it)))

            update = c_ast.ExprList([c_ast.BinaryOp('+=', it_var, c_ast.Constant('int', '1')),
                                     c_ast.Assignment('=', c_ast.ID(it), array_ref(mem, it_var.name))])
            loop.next = update
        else:
            raise TypeError("Cannot infer iterator type")


def kernel(func, specs, env=None):
    """Build OpenCL kernel source string from *func*"""
    fdef = parser.parse(func)
    params = fdef.decl.type.args.params

    # add __kernel qualifier to signature
    fdef.decl.type.type.quals = ['__kernel']

    # assign arg types
    for i, p in enumerate(params):
        if p.name in specs:
            spec = specs[p.name].qualifier
            params[i] = ptr_decl(p.name, 'float', [spec.cl_keyword])

    # create work item indices
    fdef.body.block_items.insert(0, local_decl('idx', 'int', work_item_index()))

    # create variables for intermediate results
    globalvars = [p.name for p in params]
    localvars = []
    assignments = find_type(fdef.body, c_ast.Assignment)

    for each in assignments:
        name = each.lvalue.name
        if not name in localvars and not name in globalvars:
            localvars.append(each.lvalue.name)

    for var in localvars:
        fdef.body.block_items.insert(0, local_decl(var, 'float'))

    # fix for loops
    fix_for_loops(fdef, specs)

    # replace global occurences with array access
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

    # optimization
    if env:
        optimize_depending_on_env(fdef, specs, env)

    generator = c_generator.CGenerator()
    return generator.visit(fdef)
