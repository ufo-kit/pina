import types
from .gen import make_kernel
from .qualifiers import AddressSpaceQualifier, NoQualifier


def source(*args):
    def _source(func):
        qual_args = []

        for arg in args:
            if isinstance(arg, AddressSpaceQualifier):
                qual_args.append(arg)
            elif isinstance(arg, AddressSpaceQualifier.__class__):
                # Someone passed in the class name without constructing a new
                # qualifier object (e.g. @source(Constant)), in this case we
                # assume float type and instantiate a new qualifier.
                qual_args.append(arg(float))
            else:
                qual_args.append(NoQualifier(arg))

        return make_kernel(func, qual_args)

    # The decorator was instantiated without any arguments. In this case
    # args[0] is the decorated function!
    if len(args) == 1 and isinstance(args[0], types.FunctionType):
        return make_kernel(args[0])

    return _source
