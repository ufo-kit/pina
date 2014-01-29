import pyfo.cast
from pycparser import c_ast


class OpVisitor(c_ast.NodeVisitor):
    def __init__(self, op):
        self.op = None
        self.left = None
        self.right = None
        self._op = op

    def visit_BinaryOp(self, node):
        if node.op == self._op:
            self.op = node
            self.left = node.left
            self.right = node.right
        else:
            self.visit(node.left)
            self.visit(node.right)


def constantify(fdef, specs, env):
    """Replace small read-only with constant memory"""
    params = fdef.decl.type.args.params
    readonly_params = pyfo.cast.find_read_only(fdef.body, params)

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


def substitute_mad(stmt):
    """Substitute "a * b + c" expressions  with "mad(a, b, c)"."""
    result = []

    class AddVisitor(c_ast.NodeVisitor):
        def visit_BinaryOp(self, node):
            if node.op in ('+', '-'):
                v = OpVisitor('*')
                v.visit(node.left)

                if v.left and v.right:
                    right = node.right

                    if node.op == '-':
                        # invert c to be able to use mad()
                        right = c_ast.UnaryOp('-', right)

                    result.append((node, v.left, v.right, right))

    AddVisitor().visit(stmt)

    for node, a, b, c in result:
        # TODO: check that a, b and c are of some float type
        args = c_ast.ExprList([a, b, c])
        mad = c_ast.FuncCall(c_ast.ID('mad'), args)
        pyfo.cast.replace(stmt, node, mad)


def is_pi(node):
    return isinstance(node, c_ast.ID) and node.name == 'pi'


def substitute_pi_funcs(fdef):
    """Substitute "sin/cos/tan(x * pi)" calls with sinpi/cospi/tanpi(x)"""
    funcs = ('sin', 'cos', 'tan')
    result = []

    class FuncVisitor(c_ast.NodeVisitor):
        def visit_FuncCall(self, node):
            name = node.name.name

            if name in funcs:
                arg = node.args.exprs[0]
                v = OpVisitor('*')
                v.visit(arg)

                if is_pi(v.left):
                    result.append((name, node, v.op, v.right))
                elif is_pi(v.right):
                    result.append((name, node, v.op, v.left))

            for _, c in node.children():
                self.visit(c)

    FuncVisitor().visit(fdef.body)

    for name, call, pi_op, replacement in result:
        pyfo.cast.replace(call, pi_op, replacement)
        call.name = c_ast.ID(name + 'pi')


def substitute_arcus_funcs(fdef):
    funcs = ('acos', 'asin', 'atan', 'atan2')
    result = []

    def is_func(node):
        return isinstance(node, c_ast.FuncCall) and node.name.name in funcs

    def is_eligible(node):
        if not isinstance(node, c_ast.BinaryOp):
            return False

        return (is_pi(node.left) and is_func(node.right)) or \
               (is_pi(node.right) and is_func(node.left))

    for node in pyfo.cast.find(fdef.body, is_eligible):
        call = node.left if isinstance(node.left, c_ast.FuncCall) else node.right
        call.name.name += 'pi'
        pyfo.cast.replace(fdef.body, node, call)


def level1(fdef, specs, env):
    constantify(fdef, specs, env)


def level2(fdef, specs, env):
    """Optimizations that might affect the result."""
    for stmt in fdef.body.block_items[1:]:
        substitute_mad(stmt)

    substitute_pi_funcs(fdef)
    substitute_arcus_funcs(fdef)
