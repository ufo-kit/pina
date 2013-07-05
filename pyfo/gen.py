#!/usr/bin/env python

import ast
import inspect


OP_MAP = {ast.Add: '+',
          ast.Sub: '-',
          ast.Mult: '*',
          ast.Div: '/'}


def _get_op_char(node):
    return OP_MAP[type(node.op)]


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
    def __init__(self, varc):
        self.varc = varc

    def get_fragment(self, node):
        self._fragment = ''
        self.visit(node)
        return self._fragment


class ExprGen(BaseGen):
    def __init__(self, varc):
        super(ExprGen, self).__init__(varc)

    def visit_Name(self, node):
        self._fragment += self.varc.get_var(node.id, None)

    def visit_BinOp(self, node):
        self._fragment += ExprGen(self.varc).get_fragment(node.left)
        self._fragment += ' {0} '.format(_get_op_char(node))
        self._fragment += ExprGen(self.varc).get_fragment(node.right)

    def visit_Num(self, node):
        self._fragment += str(node.n)

    def visit_Subscript(self, node):
        self._fragment += '{0}'.format(node.value.id)

        if isinstance(node.slice, ast.Index):
            tup = node.slice.value.elts
            x = ExprGen(self.varc).get_fragment(tup[0])
            y = '0'

            if len(tup) == 2:
                y = ExprGen(self.varc).get_fragment(tup[1])
            
            self._fragment += '[(_idy + ({0})) * _width + _idx + ({1})]'.format(y, x)


class BodyGen(BaseGen):
    def __init__(self, varc):
        super(BodyGen, self).__init__(varc)

    def visit_Assign(self, node):
        for target in node.targets:
            print type(target)
            if isinstance(target, ast.Name):
                if not self.varc.has(target.id):
                    self._fragment += 'float '

                location = self.varc.get_var(target.id, None)
                val_expr = ExprGen(self.varc).get_fragment(node.value)
                self._fragment += '{0} = {1};\n'.format(location, val_expr)

    def visit_AugAssign(self, node):
        self._fragment += self.get_var(node.target.id, None)
        self._fragment += ' {0}= '.format(_get_op_char(node))
        self._fragment += ExprGen(self.varc).get_fragment(node.value)
        self._fragment += ';\n'
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Return(self, node):
        expr = ExprGen(self.varc).get_fragment(node)
        self._fragment += 'output[_index] = {0};\n'.format(expr)


class FuncGen(ast.NodeVisitor):
    def __init__(self):
        super(FuncGen, self).__init__()
        self.kernel = ''

    def visit_FunctionDef(self, node):
        full_args = [arg.id for arg in node.args.args]
        full_args += ['output']
        arg_list = ', '.join(("__global float *{0}".format(name) for name in full_args))

        varc = VariableContainer()

        for arg in full_args:
            varc.global_vars.append(arg)

        gen = BodyGen(varc)

        self.kernel += '__kernel void {0}({1})\n'.format(node.name, arg_list)
        self.kernel += '{\n'
        self.kernel += 'unsigned int _width = get_global_size(0);\n'
        self.kernel += 'unsigned int _idx = get_global_id(0);\n'
        self.kernel += 'unsigned int _idy = get_global_id(1);\n'
        self.kernel += 'unsigned int _index = _idy * get_global_size(0) + _idx;\n'

        self.kernel += gen.get_fragment(node)
        self.kernel += '}'


def make_kernel(func):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    gen = FuncGen()
    gen.visit(tree)
    return gen.kernel
