import pyopencl as cl
import numpy as np

import pyfo
from ..misc import get_type_from_py


context = cl.create_some_context(False)
queue = cl.CommandQueue(context)
buffers = {}


def np_arrays(args):
    return [arg for arg in args if arg.__class__ == np.ndarray]


class JustInTimeCall(object):
    def __init__(self, func, work_size):
        self.func = pyfo.jit(func)
        self.name = func.__name__
        self.kernels = {}
        self.buffers = {}
        self.output = None

    def __call__(self, *args):
        qualified_args = [get_type_from_py(arg) for arg in args]
        key = tuple(arg.__class__.__name__ for arg in qualified_args)

        if key in self.kernels:
            kernel = self.kernels[key]
        else:
            source = self.func(*args)
            program = cl.Program(context, source).build()
            kernel = getattr(program, self.name)
            self.kernels[key] = kernel

        arrays = np_arrays(args)

        # copy arguments
        for each in arrays:
            if id(each) in self.buffers:
                buf = self.buffers[id(each)]
                cl.enqueue_copy(queue, buf, each)
            else:
                flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
                buf = cl.Buffer(context, flags, each.nbytes, hostbuf=each)
                self.buffers[id(each)] = buf

        # Create output if necessary
        if self.output is None:
            # TODO: use user-supplied information if necessary
            self.output = np.empty_like(arrays[0])
            out_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, self.output.nbytes)
            self.buffers[id(self.output)] = out_buffer
        else:
            out_buffer = self.buffers[id(self.output)]

        kargs = [self.buffers[id(each)] for each in arrays]
        kargs.append(out_buffer)
        kernel(queue, arrays[0].shape, None, *kargs)

        cl.enqueue_copy(queue, self.output, out_buffer)
        return self.output


def jit(func, work_size=None):
    return JustInTimeCall(func, work_size)
