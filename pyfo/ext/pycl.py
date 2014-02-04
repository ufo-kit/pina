import time
import pyopencl as cl
import numpy as np

import pyfo
import pyfo.cl


platform = cl.get_platforms()[0]
devices = platform.get_devices()
context = cl.Context(devices=devices)
queues = [cl.CommandQueue(context, device=d) for d in devices]
buffers = {}

env = pyfo.cl.ExecutionEnvironment()
env.MAX_CONSTANT_SIZE = min(d.max_constant_buffer_size for d in devices)
env.MAX_CONSTANT_ARGS = min(d.max_constant_args for d in devices)


def slices(array, axis, n_devices):
    rng = [dim for dim in array.shape]
    rng[axis] = rng[axis] / n_devices

    for i in range(n_devices):
        slices = []

        for j, dim in enumerate(rng):
            if j == axis:
                slices.append(slice(i * dim, (i + 1) * dim, None))
            else:
                slices.append(slice(0, dim, None))

        yield slices


class JustInTimeCall(object):

    INIT_FLAGS = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR

    def __init__(self, func, opt_level, use_multi_gpu):
        env.opt_level = opt_level
        self.func = pyfo.jit(func, env=env)
        self.use_multi_gpu = use_multi_gpu
        self.name = func.__name__
        self.buffers = {}
        self.out_buffers = {}
        self.kernel = None
        self.output = None
        self.temporary = None
        self.n_devices = len(devices)
        self.time = 0.0

    def run_multi_gpu(self, *args):
        np_args = [a for a in args if isinstance(a, np.ndarray)]
        key = tuple(id(a) for a in np_args)
        largest_shape = sorted([a.shape for a in np_args])[0]

        axis = np.argmin(largest_shape)

        kargs = []
        out_buffers = []

        for arg in args:
            if isinstance(arg, np.ndarray):
                if id(arg) in self.buffers:
                    sub_buffers = self.buffers[id(arg)]

                    for i, s in enumerate(slices(arg, axis, self.n_devices)):
                        hostbuf = np.copy(arg[s]) if axis > 0 else arg[s]
                        cl.enqueue_copy(queues[i], sub_buffers[i], hostbuf)
                else:
                    sub_buffers = []

                    for s in slices(arg, axis, self.n_devices):
                        hostbuf = np.copy(arg[s]) if axis > 0 else arg[s]
                        buf = cl.Buffer(context, self.INIT_FLAGS, 0, hostbuf=hostbuf)
                        sub_buffers.append(buf)

                    self.buffers[id(arg)] = sub_buffers

                kargs.append(sub_buffers)
            else:
                kargs.append(np.float32(arg))

        out_size = np.multiply(*largest_shape) * 4

        out_shape = [dim for dim in largest_shape]
        out_shape[axis] /= self.n_devices

        if key in self.out_buffers:
            out_buffers = self.out_buffers[key]
        else:
            for i in range(self.n_devices):
                buf_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=int(out_size))
                out_buffers.append(buf_out)

            self.out_buffers[key] = out_buffers

        start = time.time()

        for i in range(self.n_devices):
            cargs = []

            for k in kargs:
                cargs.append(k[i] if isinstance(k, list) else k)

            cargs.append(out_buffers[i])
            self.kernel(queues[i], out_shape, None, *cargs)

        if self.output is None:
            self.output = np.empty_like(arg)
            self.temporary = np.empty(out_shape).astype(np.float32)

        for i, s in enumerate(slices(arg, axis, self.n_devices)):
            if axis > 0:
                cl.enqueue_copy(queues[i], self.temporary, out_buffers[i])
                self.output[s] = self.temporary
            else:
                cl.enqueue_copy(queues[i], self.output[s], out_buffers[i])

        self.time = time.time() - start
        return self.output

    def run_single_gpu(self, shape, *args):
        kargs = []

        for arg in args:
            if isinstance(arg, np.ndarray):
                if id(arg) in self.buffers:
                    buf = self.buffers[id(arg)]
                    cl.enqueue_copy(queues[0], buf, arg)
                else:
                    flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
                    buf = cl.Buffer(context, flags, arg.nbytes, hostbuf=arg)
                    self.buffers[id(arg)] = buf

                kargs.append(buf)
            else:
                kargs.append(np.float32(arg))

        # TODO: use user-supplied information if necessary
        first_np_array = [a for a in args if isinstance(a, np.ndarray)][0]
        workspace = shape if shape else tuple([dim for dim in first_np_array.shape[::-1]])

        if self.output is None:
            self.output = np.empty(workspace).astype(np.float32)
            out_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, self.output.nbytes)
            self.buffers[id(self.output)] = out_buffer
        else:
            out_buffer = self.buffers[id(self.output)]

        kargs.append(out_buffer)

        start = time.time()
        self.kernel(queues[0], workspace, None, *kargs)
        cl.enqueue_copy(queues[0], self.output, out_buffer)
        self.time = time.time() - start
        return self.output

    def __call__(self, *args, **kwargs):
        shape = kwargs.get('shape', None)

        if not self.kernel:
            source = self.func(*args)
            program = cl.Program(context, source).build()
            self.kernel = getattr(program, self.name)

        if self.use_multi_gpu:
            return self.run_multi_gpu(*args)
        else:
            return self.run_single_gpu(shape, *args)


def jit(func, opt_level=2, use_multi_gpu=False):
    return JustInTimeCall(func, opt_level, use_multi_gpu)
