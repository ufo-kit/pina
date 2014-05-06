#!/usr/bin/env python

import numpy as np
from pina.ext.pycl import Runtime


m = Runtime()


def k_add(x, y):
    return x + y


def k_scale(s, x):
    return s * x


def k_cos(x):
    return np.cos(x)


def k_cospi(x):
    return np.cos(x * np.pi)


def k_acospi(x):
    return np.arccos(x) / np.pi


def k_complexmad(x, y):
    return 2.0 * x + (3.5 * y + x)


def k_mad_scalar(a, x, y):
    return a * x + y


def compare(func, *args):
    reference = func(*args)
    result = m.jit(func)(*args)
    assert (np.linalg.norm(reference - result) < 0.01)


class TestBasics(object):
    def setUp(self):
        self.a = np.random.random((512, 256)).astype(np.float32)
        self.b = np.random.random((512, 256)).astype(np.float32)

    def test_add(self):
        compare(k_add, self.a, self.b)

    def test_scale(self):
        compare(k_scale, 2.0, self.b)

    def test_cos(self):
        compare(k_cos, self.a)

    def test_cospi(self):
        compare(k_cospi, self.a)

    def test_acospi(self):
        compare(k_acospi, self.a)

    def test_complexmad(self):
        compare(k_complexmad, self.a, self.b)

    def test_mad_scalar(self):
        compare(k_mad_scalar, 2.0, self.a, self.b)
