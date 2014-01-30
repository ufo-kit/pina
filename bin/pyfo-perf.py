#!/usr/bin/env python

import sys
import time
import math
import numpy as np
from progress.bar import Bar
from pyfo.ext.pycl import jit, JustInTimeCall


N_ITERATIONS = 10


def saxpy_test(a, x, y):
    return a * x + y


def cos_test(x):
    return np.cos(x)


def x_pi_test(x):
    return x * np.pi


def cospi_test(x):
    return np.cos(x * np.pi)


def acospi_test(x):
    return np.arccos(x) / np.pi


def measure_call(func, *args):
    times = []

    # avoid cold start, e.g. due to kernel compilation
    func(*args)

    for i in range(N_ITERATIONS):
        start = time.time()
        func(*args)
        end = time.time()

        if isinstance(func, JustInTimeCall):
            times.append(func.time)
        else:
            times.append(end - start)


    return (np.mean(times), np.std(times))


def run_single_gpu_tests():
    opt_level = 0
    n_tests = 4
    results = {}
    widths = [1024]
    heights = list(range(1024, 8193, 512))
    bar = Bar('Measuring', max=n_tests*len(widths)*len(heights))

    for width in widths:
        for height in heights:
            x = np.random.random((width, height)).astype(np.float32)
            y = np.random.random((width, height)).astype(np.float32)

            tests = [
                (saxpy_test, (2.0, x, y)),
                (cos_test, (x,)),
                (cospi_test, (x,)),
                (acospi_test, (x,)),
            ]

            for func, args in tests:
                fname = func.__name__
                mean_np, std_np = measure_call(func, *args)
                mean_cl, std_cl = measure_call(jit(func, opt_level), *args)
                tup = (mean_np, std_np, mean_cl, std_cl)

                if fname in results:
                    results[fname][(width, height)] = tup
                else:
                    results[fname] = {(width, height): tup}

                bar.next()

    bar.finish()

    sys.stdout.write("width  height  ")

    for name in results:
        sys.stdout.write('mnp_{name}  mcl_{name}  speed_{name}  '.format(name=name))

    sys.stdout.write("\n")

    for width in widths:
        for height in heights:
            sys.stdout.write('{}  {}  '.format(width, height))

            for name in results:
                mean_np, std_np, mean_cl, std_cl = results[name][(width, height)]
                sys.stdout.write('{}  {}  {}  '.format(mean_np, mean_cl, mean_np / mean_cl))

            sys.stdout.write('\n')


if __name__ == '__main__':
    run_single_gpu_tests()
