import types
import inspect
from .gen import make_kernel
from .qualifiers import *


def get_qualified_arg(arg):
    if isinstance(arg, AddressSpaceQualifier):
        return arg

    if arg in (Global, Constant, Local):
        # Someone passed in the class name without constructing a new
        # qualifier object (e.g. @source(Constant)), in this case we
        # assume float type and instantiate a new qualifier.
        return arg(float)

    return NoQualifier(arg)


def source(*args):
    def _source(func):
        qual_args = [get_qualified_arg(arg) for arg in args]
        return make_kernel(func, qual_args)

    # The decorator was instantiated without any arguments. In this case
    # args[0] is the decorated function!
    if len(args) == 1 and isinstance(args[0], types.FunctionType):
        return make_kernel(args[0])

    return _source
