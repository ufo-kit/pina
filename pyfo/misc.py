import ast
import inspect
import collections
import types
from .gen import kernel
from .qualifiers import *



def guess_arg_types(func):
    class LvalueVisitor(ast.NodeVisitor):
        def __init__(self, secondary):
            self.secondary = secondary

        def visit_Assign(self, node):
            self.secondary.visit(node.value)

    class ArrayAccessVisitor(ast.NodeVisitor):
        def __init__(self):
            self.names = []

        def visit_Subscript(self, node):
            self.names.append(node.value.id)

    class ArgsVisitor(ast.NodeVisitor):
        def __init__(self):
            self.names = None

        def visit_FunctionDef(self, node):
            self.names = [arg.id for arg in node.args.args]

    args = ArgsVisitor()
    args.visit(func)
    accesses = ArrayAccessVisitor()
    LvalueVisitor(accesses).visit(func)

    types = []

    for arg in args.names:
        if arg in accesses.names:
            types.append(Global(float))
        else:
            types.append(None)

    return types


def get_qualified_arg(arg):
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
        qual_args = [get_qualified_arg(arg) for arg in args]
        return kernel(func, qual_args)

    # The decorator was instantiated without any arguments. In this case
    # args[0] is the decorated function!
    if len(args) == 1 and isinstance(args[0], types.FunctionType):
        return kernel(args[0])

    return _source


def get_type_from_py(arg):
    import numpy as np

    def check_supported(type_name):
        if not is_supported(type_name):
            raise RuntimeError("Unsupported data type {0}".format(type_name))

    if arg.__class__ == np.ndarray:
        check_supported(repr(arg.dtype.type))
        return Global(arg.dtype.type)
    else:
        check_supported(repr(arg.__class__))
        return NoQualifier(arg.__class__)


def invoked(func):
    def _wrapper(*args):
        if len(inspect.getargspec(func).args) != len(args):
            raise RuntimeError("Wrong arguments number.")
        qual_args = [get_type_from_py(arg) for arg in args]
        return make_kernel(func, qual_args)

    return _wrapper
