import time
import pyopencl as cl
import numpy as np

import pyfo
import pyfo.cl


context = cl.create_some_context(False)
queue = cl.CommandQueue(context)
buffers = {}

env = pyfo.cl.ExecutionEnvironment()
env.MAX_CONSTANT_SIZE = context.devices[0].max_constant_buffer_size
env.MAX_CONSTANT_ARGS = context.devices[0].max_constant_args


def np_arrays(args):
    return [arg for arg in args if arg.__class__ == np.ndarray]


class JustInTimeCall(object):
    def __init__(self, func, opt_level):
        env.opt_level = opt_level
        self.func = pyfo.jit(func, env=env)
        self.name = func.__name__
        self.buffers = {}
        self.kernel = None
        self.output = None
        self.time = 0.0

    def __call__(self, *args):
        key = tuple(id(arg) for arg in args)

        if not self.kernel:
            source = self.func(*args)
            program = cl.Program(context, source).build()
            self.kernel = getattr(program, self.name)

        kargs = []

        for arg in args:
            if isinstance(arg, np.ndarray):
                if id(arg) in self.buffers:
                    buf = self.buffers[id(arg)]
                    cl.enqueue_copy(queue, buf, arg)
                else:
                    flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
                    buf = cl.Buffer(context, flags, arg.nbytes, hostbuf=arg)
                    self.buffers[id(arg)] = buf

                kargs.append(buf)
            else:
                kargs.append(np.float32(arg))

        # TODO: use user-supplied information if necessary
        first_np_array = [a for a in args if isinstance(a, np.ndarray)][0]

        if self.output is None:
            self.output = np.empty_like(first_np_array)
            out_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, self.output.nbytes)
            self.buffers[id(self.output)] = out_buffer
        else:
            out_buffer = self.buffers[id(self.output)]

        kargs.append(out_buffer)

        start = time.time()
        self.kernel(queue, first_np_array.shape, None, *kargs)
        cl.enqueue_copy(queue, self.output, out_buffer)
        self.time = time.time() - start
        return self.output


def jit(func, opt_level=2):
    return JustInTimeCall(func, opt_level)
