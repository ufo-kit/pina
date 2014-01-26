import pyopencl as cl
import numpy as np

import pyfo
import pyfo.cl
# from ..misc import get_type_from_py


context = cl.create_some_context(False)
queue = cl.CommandQueue(context)
buffers = {}

env = pyfo.cl.ExecutionEnvironment()
env.MAX_CONSTANT_SIZE = context.devices[0].max_constant_buffer_size
env.MAX_CONSTANT_ARGS = context.devices[0].max_constant_args


def np_arrays(args):
    return [arg for arg in args if arg.__class__ == np.ndarray]


class JustInTimeCall(object):
    def __init__(self, func, work_size):
        self.func = pyfo.jit(func, env=env)
        self.name = func.__name__
        self.kernels = {}
        self.buffers = {}
        self.output = None

    def __call__(self, *args):
        key = tuple(id(arg) for arg in args)

        if key in self.kernels:
            kernel = self.kernels[key]
        else:
            source = self.func(*args)
            print source
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
                flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
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
