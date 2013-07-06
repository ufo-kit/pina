from .gen import make_kernel


def source(func):
    """Decorator that returns an OpenCL representation of the decorated
    function `func'."""
    return make_kernel(func)
