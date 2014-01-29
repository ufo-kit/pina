#!/usr/bin/env python

import numpy as np
import pyfo.mod
from pyfo import jit, ExecutionEnvironment
from pycparser import c_ast


env = ExecutionEnvironment()


@jit(env=env, ast=True)
def k_scalar(s, x):
    return s * x


class TestBasics(object):
    def setUp(self):
        self.a = np.ones((512, 512))
        self.b = np.ones((512, 512))

    def test_scalar(self):
        ast = k_scalar(3.5, self.a)
        s = pyfo.mod.find(ast, lambda node: isinstance(node, c_ast.Decl) and node.name == 's')
        assert(len(s) == 1)
        assert(isinstance(s[0].type, c_ast.TypeDecl))
        assert(isinstance(s[0].type.type, c_ast.IdentifierType))

