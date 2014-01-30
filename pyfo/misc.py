import ast
import inspect
import types
from .gen import kernel, ast
from .qualifiers import *
from .cl import BufferSpec, ExecutionEnvironment



def qualified_arg(arg):
    if isinstance(arg, AddressSpaceQualifier):
        return arg

    if arg in (Global, Constant, Local):
        # Someone passed in the class name without constructing a new
        # qualifier object (e.g. @source(Constant)), in this case we
        # assume float type and instantiate a new qualifier.
        return arg(float)

    return NoQualifier(arg)


def static(*args):
    def _source(func):
        arg_names = inspect.getargspec(func).args
        specs = {}

        for name, arg in zip(arg_names, args):
            spec = BufferSpec(name)
            spec.qualifier = qualified_arg(arg)
            specs[name] = spec

        return kernel(func, specs)

    # The decorator was instantiated without any arguments. In this case
    # args[0] is the decorated function!
    if len(args) == 1 and isinstance(args[0], types.FunctionType):
        return kernel(args[0], {})

    return _source


def arg_spec(arg, name):
    import numpy as np

    def check_supported(type_name):
        if not is_supported(type_name):
            raise RuntimeError("Unsupported data type {0}".format(type_name))

    spec = BufferSpec(name)

    if arg.__class__ == np.ndarray:
        check_supported(repr(arg.dtype.type))
        spec.size = arg.nbytes
        spec.qualifier = Global(arg.dtype.type)
    else:
        check_supported(repr(arg.__class__))
        spec.qualifier = NoQualifier(arg.__class__)

    return spec


class jit(object):
    def __init__(self, *args, **kwargs):
        self.env = kwargs.get('env', None)
        self.return_ast = kwargs.get('ast', False)
        self.func = args[0] if args else None

    def __call__(self, *cargs):
        if not self.func:
            self.func = cargs[0]

        def _wrapper(*args):
            arg_names = inspect.getargspec(self.func).args
            num_expected = len(arg_names)

            if num_expected != len(args):
                msg = "{}() takes exactly {} arguments ({} given)"
                raise TypeError(msg.format(self.func.__name__, num_expected, len(args)))

            specs = {name: arg_spec(a, name) for a, name in zip(args, arg_names)}

            if not self.return_ast:
                return kernel(self.func, specs, env=self.env)

            return ast(self.func, specs, env=self.env)

        if not isinstance(cargs[0], types.FunctionType):
            return _wrapper(*cargs)

        return _wrapper
