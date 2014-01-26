import ast
import inspect
from pycparser import c_ast


class PythonToC(ast.NodeVisitor):
    def __init__(self):
        self.result = None

    def visit_Return(self, node):
        self.result = c_ast.Return(python_to_c_ast(node.value))

    def visit_BinOp(self, node):
        self.result = c_ast.BinaryOp(python_to_c_ast(node.op),
                                     python_to_c_ast(node.left),
                                     python_to_c_ast(node.right))

    def visit_If(self, node):
        else_branch = None if not node.orelse else python_to_c_ast(node.orelse)
        self.result = c_ast.If(python_to_c_ast(node.test),
                               python_to_c_ast(node.body),
                               else_branch)

    def visit_Compare(self, node):
        self.result = c_ast.BinaryOp(python_to_c_ast(node.ops[0]),
                                     python_to_c_ast(node.left),
                                     python_to_c_ast(node.comparators[0]))

    def visit_Name(self, node):
        self.result = c_ast.ID(node.id)

    def generic_visit(self, node):
        ops = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
               ast.Gt: '>', ast.GtE: '>=', ast.Lt: '<', ast.LtE: '<=',
               ast.NotEq: '!=',
               ast.And: '&&', ast.Or: '||'}

        if type(node) in ops:
            self.result = ops[type(node)]
        else:
            for _, c in node.children():
                self.visit(c)

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
    node = tree.body[0]

    # Convert statements in the body
    stmts = [python_to_c_ast(stmt) for stmt in node.body]
    body = c_ast.Compound(stmts) 

    # Build function declaration
    c_args = [python_to_c_ast(arg) for arg in node.args.args]
    tdecl = c_ast.TypeDecl(func.__name__, None, c_ast.IdentifierType(['void']))
    fdecl = c_ast.FuncDecl(c_ast.ParamList(c_args), tdecl)
    decl = c_ast.Decl(func.__name__, None, None, None, None, fdecl, None, None)
    fdef = c_ast.FuncDef(decl, None, body, None)

    return fdef
