import types
from .gen import make_kernel
from .qualifiers import AddressSpaceQualifier, NoQualifier


def source(*args):
    def _source(func):
        qual_args = [arg if isinstance(arg, AddressSpaceQualifier)
                     else NoQualifier(arg)
                     for arg in args]
        return make_kernel(func, qual_args)

    if len(args) == 1 and isinstance(args[0], types.FunctionType):
        return make_kernel(args[0])

    return _source
