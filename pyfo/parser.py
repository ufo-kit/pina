import ast
import inspect
from pycparser import c_ast


def create_decl(name, typename, init):
    idtype = c_ast.IdentifierType([typename])
    typedecl = c_ast.TypeDecl(name, [], idtype)
    return c_ast.Decl(name, None, None, None, None, typedecl, init, None)


def constant(name):
    """
    Check if *name* is a known constant and return an equivalent
    c_ast.Constant(). If it is unknown, return None.
    """
    known = ('pi', 'e', 'ln2', 'ln10')

    if name in known:
        return c_ast.Constant('float', name)

    return None


class PythonToC(ast.NodeVisitor):
    def __init__(self):
        self.result = None

    def generic_visit(self, node):
        ops = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
               ast.Gt: '>', ast.GtE: '>=', ast.Lt: '<', ast.LtE: '<=',
               ast.NotEq: '!=',
               ast.And: '&&', ast.Or: '||',
               ast.Invert: '~', ast.Not: '!', ast.UAdd: '+', ast.USub: '-'}

        if type(node) in ops:
            self.result = ops[type(node)]
        else:
            for _, c in node.children():
                self.visit(c)

    def visit_Return(self, node):
        self.result = c_ast.Return(python_to_c_ast(node.value))

    def visit_UnaryOp(self, node):
        self.result = c_ast.UnaryOp(python_to_c_ast(node.op),
                                    python_to_c_ast(node.operand))

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            self.result = c_ast.FuncCall(c_ast.ID('pow'),
                                         c_ast.ExprList([python_to_c_ast(node.left),
                                                         python_to_c_ast(node.right)]))
        else:
            self.result = c_ast.BinaryOp(python_to_c_ast(node.op),
                                         python_to_c_ast(node.left),
                                         python_to_c_ast(node.right))

    def visit_If(self, node):
        else_branch = None if not node.orelse else python_to_c_ast(node.orelse)
        self.result = c_ast.If(python_to_c_ast(node.test),
                               python_to_c_ast(node.body),
                               else_branch)

    def visit_For(self, node):
        if isinstance(node.iter, ast.Name):
            # We are iterating over a named iterable which we will convert to
            # an array iteration
            body = python_to_c_ast(node.body)
            self.result = c_ast.For(None, None, None, body)

            # Mark this for-loop to be processed once input data is known
            setattr(self.result, '_extra', (node.target.id, node.iter.id))
        elif isinstance(node.iter, ast.Call) and node.iter.func.id == 'range':
            var = node.target.id
            args = [python_to_c_ast(arg) for arg in node.iter.args]
            frm = c_ast.Constant('int', '0') if len(args) == 1 else args[0]
            to = args[0] if len(args) == 1 else args[1]
            step = c_ast.Constant('int', '1') if len(args) <= 2 else args[2]

            init = create_decl(var, 'int', frm)
            cond = c_ast.BinaryOp('<', c_ast.ID(var), to)
            update = c_ast.ExprList([c_ast.BinaryOp('+=', c_ast.ID(var), step)])

            self.result = c_ast.For(init, cond, update, python_to_c_ast(node.body))

    def visit_Compare(self, node):
        self.result = c_ast.BinaryOp(python_to_c_ast(node.ops[0]),
                                     python_to_c_ast(node.left),
                                     python_to_c_ast(node.comparators[0]))

    def visit_Name(self, node):
        self.result = c_ast.ID(node.id)

    def visit_Num(self, node):
        self.result = c_ast.Constant('int', str(node.n))

    def visit_Index(self, node):
        self.result = python_to_c_ast(node.value)

    def visit_Subscript(self, node):
        self.result = c_ast.ArrayRef(python_to_c_ast(node.value),
                                     python_to_c_ast(node.slice))

    def visit_Assign(self, node):
        lvalue = python_to_c_ast(node.targets[0])
        rvalue =  python_to_c_ast(node.value)
        self.result = c_ast.Assignment('=', lvalue, rvalue)

    def visit_AugAssign(self, node):
        op = python_to_c_ast(node.op)
        lvalue = python_to_c_ast(node.target)
        rvalue =  python_to_c_ast(node.value)
        self.result = c_ast.Assignment(op + '=', lvalue, rvalue)

    def visit_Tuple(self, node):
        elements = [python_to_c_ast(e) for e in node.elts]

        def add_ops(lst, op):
            if not lst:
                return op

            op = c_ast.BinaryOp('+', op, lst[0])
            return add_ops(lst[1:], op)

        start = c_ast.BinaryOp('+', elements[0], elements[1])
        self.result = add_ops(elements[2:], start)

    def visit_Attribute(self, node):
        # strip off attribute accesses
        self.result = constant(node.attr)

        if not self.result:
            self.result = c_ast.ID(node.attr)

    def visit_Call(self, node):
        # TODO: check if call to an OpenCL function and leave it ...
        args = [python_to_c_ast(arg) for arg in node.args]
        self.result = c_ast.FuncCall(python_to_c_ast(node.func), c_ast.ExprList(args))

        # ... otherwise we will flatten later

    def visit_FunctionDef(self, node):
        params = [python_to_c_ast(arg) for arg in node.args.args]
        rtype = c_ast.IdentifierType(['void'])
        tdecl = c_ast.TypeDecl(node.name, ['__kernel'], rtype)
        ftype = c_ast.FuncDecl(c_ast.ParamList(params), tdecl)
        decl = c_ast.Decl(node.name, None, None, None, ftype, None, None)
        body = python_to_c_ast(node.body)
        self.result = c_ast.FuncDef(decl, None, body)


def python_to_c_ast(py_node):
    if isinstance(py_node, list):
        body = [python_to_c_ast(stmt) for stmt in py_node]
        return c_ast.Compound(body)
    else:
        v = PythonToC()
        v.visit(py_node)
        return v.result


def parse(func):
    """
    Turn *func*'s Python AST into a pycparser AST for subsequent
    optimization and OpenCL code generation.
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    return python_to_c_ast(tree.body[0])
