import pyfo.mod
from pycparser import c_ast


class MulVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.op = None
        self.left = None
        self.right = None

    def visit_BinaryOp(self, node):
        if node.op == '*':
            self.op = node
            self.left = node.left
            self.right = node.right
        else:
            self.visit(node.left)
            self.visit(node.right)


def constantify(fdef, specs, env):
    """Replace small read-only with constant memory"""
    params = fdef.decl.type.args.params
    readonly_params = pyfo.mod.find_read_only(fdef.body, params)

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
        pyfo.mod.replace(fdef, node, mad)


def substitute_pi_funcs(fdef, specs, env):
    """Substitute "sin/cos/tan(x * pi)" calls with sinpi/cospi/tanpi(x)"""
    funcs = ('sin', 'cos', 'tan')
    result = []

    def is_pi(node):
        return isinstance(node, c_ast.ID) and node.name == 'PI'

    class FuncVisitor(c_ast.NodeVisitor):
        def visit_FuncCall(self, node):
            name = node.name.name

            if name in funcs:
                arg = node.args.exprs[0]
                v = MulVisitor()
                v.visit(arg)

                if is_pi(v.left):
                    result.append((name, node, v.op, v.right))
                elif is_pi(v.right):
                    result.append((name, node, v.op, v.left))

    FuncVisitor().visit(fdef.body)

    for name, call, pi_op, replacement in result:
        pyfo.mod.replace(call, pi_op, replacement)
        call.name = c_ast.ID(name + 'pi')


def level1(fdef, specs, env):
    constantify(fdef, specs, env)


def level2(fdef, specs, env):
    """Optimizations that might affect the result."""
    substitute_mad(fdef, specs, env)
    substitute_pi_funcs(fdef, specs, env)
