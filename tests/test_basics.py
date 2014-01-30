#!/usr/bin/env python

import numpy as np
import pyfo.cast
from pyfo import jit, ExecutionEnvironment
from pycparser import c_ast


env = ExecutionEnvironment()


@jit(env=env, ast=True)
def k_scalar(s, x):
    return s * x


@jit(ast=True)
def k_relative(x):
    return 0.5 * x[-1] + 0.5 * x[+1]


@jit(ast=True)
def k_rename(x):
    return np.arctan(x)


class TestBasics(object):
    def setUp(self):
        self.a = np.ones((512, 512))
        self.b = np.ones((512, 512))

    def test_scalar(self):
        ast = k_scalar(3.5, self.a)
        s = pyfo.cast.find(ast, lambda node: isinstance(node, c_ast.Decl) and node.name == 's')
        assert(len(s) == 1)
        assert(isinstance(s[0].type, c_ast.TypeDecl))
        assert(isinstance(s[0].type.type, c_ast.IdentifierType))

    def test_relative(self):
        ast = k_relative(self.a)

    def test_rename(self):
        ast = k_rename(self.a)
        calls = pyfo.cast.find_type(ast, c_ast.FuncCall)
        assert(len(calls) == 1)

        call = calls[0]
        assert(call.name.name == 'atan')
