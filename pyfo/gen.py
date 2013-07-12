#!/usr/bin/env python

import sys

PYTHON_3 = sys.version_info >= (3, 0)

import ast
import inspect

from itertools import izip_longest
from .qualifiers import NoQualifier, Global


OP_MAP = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.And: '&&',
    ast.Or: '||',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.NotEq: '!=',
}


def _get_op_char(node):
    return OP_MAP[type(node)]


class Variable(object):
    def __init__(self, name, qualifier=None):
        self.name = name
        self.qualifier = qualifier if qualifier else NoQualifier(float)

    def declaration(self):
        asterisk = '' if isinstance(self.qualifier, NoQualifier) else '*'
        return '{0} {1} {2}{3}'.format(self.qualifier.cl_keyword,
                                       self.qualifier.type_name,
                                       asterisk,
                                       self.name)

    def fragment(self, expr=None):
        if isinstance(self.qualifier, NoQualifier):
            return self.name

        if expr:
            return '{0}[{1}]'.format(self.name, expr.fragment())

        return '{0}[_index]'.format(self.name)


def get_var(d, name):
    if not name in d:
        d[name] = Variable(name, NoQualifier('float'))

    return d[name]


class BaseGen(ast.NodeVisitor):

    indentation = ''

    def __init__(self, varc):
        self.varc = varc

    def fragment(self, node):
        self._fragment = ''
        self.visit(node)
        return self._fragment

    def add(self, fragment):
        self._fragment += self.indentation + fragment

    def indent(self):
        self.indentation += '    '

    def unindent(self):
        self.indentation = self.indentation[:-4]


class ExprGen(BaseGen):
    def __init__(self, varc):
        super(ExprGen, self).__init__(varc)

    def visit_Name(self, node):
        self.add(get_var(self.varc, node.id).fragment())

    def visit_BinOp(self, node):
        self.add(ExprGen(self.varc).fragment(node.left))
        self.add(' {0} '.format(_get_op_char(node.op)))
        self.add(ExprGen(self.varc).fragment(node.right))

    def visit_BoolOp(self, node):
        self.add(ExprGen(self.varc).fragment(node.values[0]))
        self.add(' {0} '.format(_get_op_char(node.op)))
        self.add(ExprGen(self.varc).fragment(node.values[1]))

    def visit_Compare(self, node):
        def gen_comparisons():
            left = node.left

            for op, comparator in zip(node.ops, node.comparators):
                s = ExprGen(self.varc).fragment(left)
                s += ' {0} '.format(_get_op_char(op))
                s += ExprGen(self.varc).fragment(comparator)
                left = comparator
                yield s

        self.add(' && '.join('({0})'.format(c) for c in gen_comparisons()))

    def visit_Num(self, node):
        self.add(str(node.n))

    def visit_Subscript(self, node):
        self.add('{0}'.format(node.value.id))

        if isinstance(node.slice, ast.Index):
            if isinstance(node.slice.value, ast.Tuple):
                tup = node.slice.value.elts
                x = ExprGen(self.varc).fragment(tup[0])
                y = ExprGen(self.varc).fragment(tup[1])
                self.add('[(_idy + ({0})) * _width + _idx + ({1})]'.format(y, x))
            else:
                index = ExprGen(self.varc).fragment(node.slice.value)
                self.add('[{0}]'.format(index))

    def visit_IfExp(self, node):
        self.add(ExprGen(self.varc).fragment(node.test))
        self.add(' ?  {0}'.format(ExprGen(self.varc).fragment(node.body)))
        self.add(' : {0};'.format(ExprGen(self.varc).fragment(node.orelse)))

    def visit_Call(self, node):
        self.add(node.func.id + '(')
        self.add(', '.join(ExprGen(self.varc).fragment(arg)
                           for arg in node.args))
        self.add(')')


class GenericVisitor(ast.NodeVisitor):
    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)


def get_names(expr):
    class Visitor(GenericVisitor):
        def __init__(self):
            self.names = []

        def visit_Name(self, node):
            self.names.append(node.id)

    v = Visitor()
    v.visit(expr)
    return v.names


def has_return_stmt(node):
    class Visitor(GenericVisitor):
        def __init__(self):
            self.has_return = False

        def visit_Return(self, node):
            self.has_return = True

    v = Visitor()
    v.visit(node)
    return v.has_return


class StmtGen(BaseGen):
    def __init__(self, varc):
        super(StmtGen, self).__init__(varc)

    def visit_Return(self, node):
        expr = ExprGen(self.varc).fragment(node)
        self.add('_output[_index] = {0};\n'.format(expr))

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                if not target.id in self.varc:
                    self.add('float ')

                location = get_var(self.varc, target.id).fragment()
                val_expr = ExprGen(self.varc).fragment(node.value)
                self.add('{0} = {1};\n'.format(location, val_expr))

    def visit_AugAssign(self, node):
        self.add(self.varc.get_var(node.target.id, None))
        self.add(' {0}= '.format(_get_op_char(node.op)))
        self.add(ExprGen(self.varc).fragment(node.value))
        self.add(';\n')

    def visit_If(self, node):
        def visit_body(body_node):
            self.indent()
            for stmt in body_node:
                self.add(StmtGen(self.varc).fragment(stmt))
            self.unindent()

        test = ExprGen(self.varc).fragment(node.test)
        self.add('if (%s) {\n' % test)
        visit_body(node.body)
        self.add('}\n')

        if len(node.orelse) > 0:
            self.add('else {\n')
            visit_body(node.orelse)
            self.add('}\n')


def type_args(args, arg_types):
    for arg, arg_type in izip_longest(args, arg_types, fillvalue=None):
        if arg:
            name = arg.arg if PYTHON_3 else arg.id
            yield (name, Variable(name, arg_type))


def argument_names(args, has_output):
    for arg in args:
        yield arg.arg if PYTHON_3 else arg.id

    if has_output:
        yield '_output'


class FuncGen(ast.NodeVisitor):
    def __init__(self, arg_types):
        super(FuncGen, self).__init__()
        self.kernel = ''
        self.arg_types = arg_types

    def visit_FunctionDef(self, node):
        has_return = has_return_stmt(node)
        varc = dict(type_args(node.args.args, self.arg_types))

        if has_return:
            varc['_output'] = Variable('_output', Global(float))

        arg_list = ', '.join(get_var(varc, name).declaration()
                             for name in argument_names(node.args.args,
                                                        has_return))

        gen = StmtGen(varc)

        self.kernel += '__kernel void {0}({1})\n'.format(node.name, arg_list)
        self.kernel += '{\n'
        self.kernel += 'unsigned int _width = get_global_size(0);\n'
        self.kernel += 'unsigned int _idx = get_global_id(0);\n'
        self.kernel += 'unsigned int _idy = get_global_id(1);\n'
        self.kernel += 'unsigned int _index = _idy * _width + _idx;\n'
        self.kernel += gen.fragment(node)
        self.kernel += '}'


def make_kernel(func, arg_types=None):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    gen = FuncGen(arg_types)
    gen.visit(tree)
    return gen.kernel
