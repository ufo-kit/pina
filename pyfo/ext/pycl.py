import sys
import time
import pyopencl as cl
import numpy as np
import pyfo
import pyfo.cl


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

    def __init__(self, func, mojito):
        self.func = pyfo.jit(func, env=mojito.env)
        self.mojito = mojito
        self.name = func.__name__
        self.buffers = {}
        self.out_buffers = {}
        self.kernel = None
        self.output = None
        self.temporary = None
        self.time = 0.0

    def __call__(self, *args, **kwargs):
        shape = kwargs.get('shape', None)

        if not self.kernel:
            source = self.func(*args)
            program = cl.Program(self.mojito.context, source).build()
            self.kernel = getattr(program, self.name)

        return self.run(self.kernel, shape, *args)

    def run(self, kernel, shape, *args):
        raise NotImplementedError


class MultiCall(JustInTimeCall):
    def __init__(self, func, mojito):
        super(MultiCall, self).__init__(func, mojito)

    def run(self, kernel, shape, *args):
        np_args = [a for a in args if isinstance(a, np.ndarray) and len(a.shape) > 1]
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
                        cl.enqueue_copy(self.mojito.queues[i], sub_buffers[i], hostbuf)
                else:
                    sub_buffers = []

                    for s in slices(arg, axis, self.mojito.n_devices):
                        hostbuf = np.copy(arg[s]) if axis > 0 else arg[s]
                        buf = cl.Buffer(self.mojito.context, self.INIT_FLAGS, 0, hostbuf=hostbuf)
                        sub_buffers.append(buf)

                    self.buffers[id(arg)] = sub_buffers

                kargs.append(sub_buffers)
            else:
                kargs.append(np.float32(arg))

        out_size = np.multiply(*largest_shape) * 4

        out_shape = [dim for dim in largest_shape]
        out_shape[axis] /= self.mojito.n_devices

        if key in self.out_buffers:
            out_buffers = self.out_buffers[key]
        else:
            for i in range(self.mojito.n_devices):
                buf_out = cl.Buffer(self.mojito.context, cl.mem_flags.WRITE_ONLY, size=int(out_size))
                out_buffers.append(buf_out)

            self.out_buffers[key] = out_buffers

        start = time.time()

        for i in range(self.mojito.n_devices):
            cargs = []

            for k in kargs:
                cargs.append(k[i] if isinstance(k, list) else k)

            cargs.append(out_buffers[i])
            kernel(self.mojito.queues[i], out_shape, None, *cargs)

        if self.output is None:
            self.output = np.empty_like(arg)
            self.temporary = np.empty(out_shape).astype(np.float32)

        for i, s in enumerate(slices(arg, axis, self.mojito.n_devices)):
            if axis > 0:
                cl.enqueue_copy(self.mojito.queues[i], self.temporary, out_buffers[i])
                self.output[s] = self.temporary
            else:
                cl.enqueue_copy(self.mojito.queues[i], self.output[s], out_buffers[i])

        self.time = time.time() - start
        return self.output


class SingleCall(JustInTimeCall):
    def __init__(self, func, mojito):
        super(SingleCall, self).__init__(func, mojito)

    def run(self, kernel, shape, *args):
        kargs = []

        for arg in args:
            if isinstance(arg, np.ndarray):
                if id(arg) in self.buffers:
                    buf = self.buffers[id(arg)]
                    cl.enqueue_copy(self.mojito.queues[0], buf, arg)
                else:
                    flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
                    buf = cl.Buffer(self.mojito.context, flags, arg.nbytes, hostbuf=arg)
                    self.buffers[id(arg)] = buf

                kargs.append(buf)
            else:
                kargs.append(np.float32(arg))

        # TODO: use user-supplied information if necessary
        first_np_array = [a for a in args if isinstance(a, np.ndarray)][0]
        workspace = shape if shape else first_np_array.shape

        if self.output is None:
            self.output = np.empty(workspace).astype(np.float32)
            out_buffer = cl.Buffer(self.mojito.context, cl.mem_flags.WRITE_ONLY, self.output.nbytes)
            self.buffers[id(self.output)] = out_buffer
        else:
            out_buffer = self.buffers[id(self.output)]

        kargs.append(out_buffer)

        start = time.time()
        kernel(self.mojito.queues[0], workspace, None, *kargs)
        cl.enqueue_copy(self.mojito.queues[0], self.output, out_buffer)
        self.time = time.time() - start
        return self.output


class Mojito(object):
    def __init__(self, opt_level=2, use_multi_gpu=False, preferred=None):
        platforms = cl.get_platforms()

        if preferred:
            needle = preferred.lower()
            candidates = [p for p in platforms
                          if needle in p.get_info(cl.platform_info.NAME).lower()]

            if candidates:
                self.platform = candidates[0]
            else:
                msg = 'Could not find preferred platform, falling back to the first one.'
                sys.stderr.write(msg)
                self.platform = platforms[0]
        else:
            self.platform = platforms[0]

        self.devices = self.platform.get_devices()
        self.context = cl.Context(devices=self.devices)
        self.queues = [cl.CommandQueue(self.context, device=d) for d in self.devices]

        self.env = pyfo.cl.ExecutionEnvironment()
        self.env.MAX_CONSTANT_SIZE = min(d.max_constant_buffer_size for d in self.devices)
        self.env.MAX_CONSTANT_ARGS = min(d.max_constant_args for d in self.devices)

        self.opt_level = opt_level
        self.use_multi_gpu = use_multi_gpu
        self.n_devices = len(self.devices)

    def jit(self, func):
        if self.use_multi_gpu:
            return MultiCall(func, self)

        return SingleCall(func, self)
