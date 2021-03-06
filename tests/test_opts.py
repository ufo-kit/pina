#!/usr/bin/env python

import numpy as np
import pina.cast
from pina import jit, ExecutionEnvironment
from pycparser import c_ast


env = ExecutionEnvironment()
env.opt_level = 2


@jit(env=env, ast=True)
def k_cospi(x, y):
    return cos(x * pi)


@jit(env=env, ast=True)
def k_acospi(x):
    return arccos(x) / pi


@jit(env=env, ast=True)
def k_mad(x, y):
    return 2 * x + y


class TestOptimizations(object):
    def setUp(self):
        self.a = np.ones((512, 512))
        self.b = np.ones((512, 512))

    def test_cospi(self):
        ast = k_cospi(self.a, self.b)
        r = pina.cast.find_type(ast, c_ast.FuncCall)
        assert len(r) == 1
        assert r[0].name.name == 'cospi'

    def test_acospi(self):
        ast = k_acospi(self.a)
        r = pina.cast.find_type(ast, c_ast.FuncCall)
        assert len(r) == 1
        assert r[0].name.name == 'acospi'

    def test_mad(self):
        ast = k_mad(self.a, self.b)
        r = pina.cast.find_type(ast, c_ast.FuncCall)
        assert len(r) == 1
        assert r[0].name.name == 'mad'
