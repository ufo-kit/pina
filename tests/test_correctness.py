#!/usr/bin/env python

import numpy as np
from pyfo.ext.pycl import jit


def k_add(x, y):
    return x + y


def k_cos(x):
    return np.cos(x)


def k_cospi(x):
    return np.cos(x * np.pi)


def k_complexmad(x, y):
    return 2.0 * x + (3.5 * y + x)


def compare(func, *args):
    reference = func(*args)
    result = jit(func)(*args)
    assert (np.linalg.norm(reference - result) < 0.01)


class TestBasics(object):
    def setUp(self):
        self.a = np.random.random((512, 256)).astype(np.float32)
        self.b = np.random.random((512, 256)).astype(np.float32)

    def test_add(self):
        compare(k_add, self.a, self.b)

    def test_cos(self):
        compare(k_cos, self.a)

    def test_cospi(self):
        compare(k_cospi, self.a)

    def test_complexmad(self):
        compare(k_complexmad, self.a, self.b)
