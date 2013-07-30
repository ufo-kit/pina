import inspect
import pyopencl as cl
import numpy as np

from ..gen import Kernel
from ..misc import get_type_from_py


context = cl.create_some_context(False)
queue = cl.CommandQueue(context)
buffers = {}


def _filter_numpy_arrays(args):
    return (arg for arg in args if arg.__class__ == np.ndarray)


class _Invocation(object):
    def __init__(self, func):
        self.func = func
        self.kernels = {}
        self.output = None
        self.returns = False

    def __call__(self, *args):
        if len(inspect.getargspec(self.func).args) != len(args):
            raise RuntimeError("Wrong number of arguments.")

        qual_args = [get_type_from_py(arg) for arg in args]
        args_key = tuple(arg.__class__.__name__ for arg in qual_args)

        if args_key in self.kernels:
            kernel = self.kernels[args_key]
        else:
            pyfo_kernel = Kernel(self.func, qual_args)
            program = cl.Program(context, pyfo_kernel.source).build()
            kernel = getattr(program, self.func.__name__)
            self.kernels[args_key] = kernel
            self.returns = pyfo_kernel.returns

        # 1. Copy all args to the device
        self.copy_to_device(args)

        first_np_array = list(_filter_numpy_arrays(args))[0]

        # 2. NOW update the args tuple, for obvious reasons we don't want to
        # copy the output array from host to device
        if self.returns:
            if self.output is None:
                self.output = np.empty_like(first_np_array)
                self.insert_buffer(self.output, cl.mem_flags.WRITE_ONLY)

            args += (self.output, )

        kernel_args = []

        for arg in args:
            if arg.__class__ == np.ndarray:
                kernel_args.append(self.get_buffer(arg))
            else:
                kernel_args.append(arg)

        kernel(queue, first_np_array.shape, None, *kernel_args)

        if self.returns:
            cl.enqueue_copy(queue, self.output, self.get_buffer(self.output))
            return self.output

    def has_buffer(self, array):
        return id(array) in buffers

    def insert_buffer(self, array, flags, **kwargs):
        key = id(array)
        mf = cl.mem_flags
        buffers[key] = cl.Buffer(context, flags, array.nbytes, **kwargs)

    def get_buffer(self, array):
        key = id(array)
        return buffers[key]

    def copy_to_device(self, args):
        for arg in _filter_numpy_arrays(args):
            if self.has_buffer(arg):
                buf = self.get_buffer(arg)
                cl.enqueue_copy(queue, buf, arg)
            else:
                flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
                self.insert_buffer(arg, flags, hostbuf=arg)


def invoked(func):
    return _Invocation(func)
