#!/usr/bin/env python

import sys

PYTHON_3 = sys.version_info >= (3, 0)

import ast
import inspect


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


class VariableContainer(object):
    def __init__(self):
        self.global_vars = []
        self.local_vars = []

    def has(self, var):
        return var in self.global_vars or var in self.local_vars

    def get_var(self, var, expr=None):
        if var in self.global_vars:
            if expr:
                return '{0}[{1}]'.format(var, expr.get_fragment())
            else:
                return '{0}[_index]'.format(var)

        self.local_vars.append(var)
        return var


class BaseGen(ast.NodeVisitor):

    indentation = ''

    def __init__(self, varc):
        self.varc = varc

    def get_fragment(self, node):
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
        self.add(self.varc.get_var(node.id, None))

    def visit_BinOp(self, node):
        self.add(ExprGen(self.varc).get_fragment(node.left))
        self.add(' {0} '.format(_get_op_char(node.op)))
        self.add(ExprGen(self.varc).get_fragment(node.right))

    def visit_BoolOp(self, node):
        self.add(ExprGen(self.varc).get_fragment(node.values[0]))
        self.add(' {0} '.format(_get_op_char(node.op)))
        self.add(ExprGen(self.varc).get_fragment(node.values[1]))

    def visit_Compare(self, node):
        def gen_comparisons():
            left = node.left

            for op, comparator in zip(node.ops, node.comparators):
                s = ExprGen(self.varc).get_fragment(left)
                s += ' {0} '.format(_get_op_char(op))
                s += ExprGen(self.varc).get_fragment(comparator)
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
                x = ExprGen(self.varc).get_fragment(tup[0])
                y = ExprGen(self.varc).get_fragment(tup[1])
                self.add('[(_idy + ({0})) * _width + _idx + ({1})]'.format(y, x))
            else:
                index = ExprGen(self.varc).get_fragment(node.slice.value)
                self.add('[{0}]'.format(index))

    def visit_IfExp(self, node):
        self.add(ExprGen(self.varc).get_fragment(node.test))
        self.add(' ?  {0}'.format(ExprGen(self.varc).get_fragment(node.body)))
        self.add(' : {0};'.format(ExprGen(self.varc).get_fragment(node.orelse)))

    def visit_Call(self, node):
        self.add(node.func.id + '(')
        self.add(', '.join(ExprGen(self.varc).get_fragment(arg) for arg in node.args))
        self.add(')')


class StmtGen(BaseGen):
    def __init__(self, varc):
        super(StmtGen, self).__init__(varc)

    def visit_Return(self, node):
        expr = ExprGen(self.varc).get_fragment(node)
        self.add('output[_index] = {0};\n'.format(expr))

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                if not self.varc.has(target.id):
                    self.add('float ')

                location = self.varc.get_var(target.id, None)
                val_expr = ExprGen(self.varc).get_fragment(node.value)
                self.add('{0} = {1};\n'.format(location, val_expr))

    def visit_AugAssign(self, node):
        self.add(self.varc.get_var(node.target.id, None))
        self.add(' {0}= '.format(_get_op_char(node.op)))
        self.add(ExprGen(self.varc).get_fragment(node.value))
        self.add(';\n')

    def visit_If(self, node):

        def visit_body(body_node):
            self.indent()
            for stmt in body_node:
                self.add(StmtGen(self.varc).get_fragment(stmt))
            self.unindent()

        test = ExprGen(self.varc).get_fragment(node.test)
        self.add('if (%s) {\n' % test)
        visit_body(node.body)
        self.add('}\n')

        if len(node.orelse) > 0:
            self.add('else {\n')
            visit_body(node.orelse)
            self.add('}\n')



def get_typed_arguments(args, arg_types):
    num_types = len(arg_types) if arg_types else 0

    for i, arg in enumerate(args):
        arg_name = arg.arg if PYTHON_3 else arg.id

        if PYTHON_3:
            if isinstance(arg.annotation, ast.Name):
                yield '{0} {1}'.format(arg.annotation.id, arg_name)
            elif isinstance(arg.annotation, ast.List):
                # DANGEROUS: assumes "elts" has elements and elts[0] is of type
                # ast.Name!
                yield '{0} *{1}'.format(arg.annotation.elts[0].id, arg_name)
            else:
                yield '__global float *{0}'.format(arg_name)
        else:
            if i >= num_types:
                # No type found
                yield '__global float *{0}'.format(arg_name)
            else:
                t = arg_types[i]
                yield '{0} {1} *{2}'.format(t.qual, t.type_name, arg_name)


def get_argument_names(args):
    if PYTHON_3:
        return [arg.arg for arg in args] + ['output']
    else:
        return [arg.id for arg in args] + ['output']


class FuncGen(ast.NodeVisitor):
    def __init__(self, arg_types):
        super(FuncGen, self).__init__()
        self.kernel = ''
        self.arg_types = arg_types

    def visit_FunctionDef(self, node):
        arg_list = ', '.join(get_typed_arguments(node.args.args, self.arg_types))
        varc = VariableContainer()
        varc.global_vars.extend(get_argument_names(node.args.args))

        gen = StmtGen(varc)

        self.kernel += '__kernel void {0}({1})\n'.format(node.name, arg_list)
        self.kernel += '{\n'
        self.kernel += 'unsigned int _width = get_global_size(0);\n'
        self.kernel += 'unsigned int _idx = get_global_id(0);\n'
        self.kernel += 'unsigned int _idy = get_global_id(1);\n'
        self.kernel += 'unsigned int _index = _idy * _width + _idx;\n'
        self.kernel += gen.get_fragment(node)
        self.kernel += '}'


def make_kernel(func, arg_types=None):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    gen = FuncGen(arg_types)
    gen.visit(tree)
    return gen.kernel
