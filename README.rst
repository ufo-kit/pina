Usage
=====

Import the source decorator to turn any Python function to OpenCL C::

    from pyfo import source

    @source
    def add(x, y):
        return 2.0 * cos(x) + sin(y)

Any return statement will generate an implicit write to the hidden ``output``
kernel argument.


Indexing
--------

Omitting square brackets reading and writing values will be local to the
executed kernel. By using square brackets, adjacent data can be accessed::

    @source
    def foo(x):
        left = x[-1]
        right = x[+1]
        top_left = x[-1, -1]
        bottom_left = x[-1, +1]


Type annotations
----------------

Static type annotations are written as simple decorator arguments. All integral
builtin types as well as all NumPy types are supported::

    @source(float, np.uint8)
    def bar(x, y):
        pass

OpenCL address space qualifiers can be specified by using the appropriate
qualifier classes::

    from pyfo import Global, Constant
    
    @source(Global(np.float32), Constant)
    def qux(x, y):
        pass


Unsupported Python constructs
-----------------------------

Some Python features cannot be mapped reasonably onto OpenCL, e.g.  ``import``,
``del``, ``lambda``, ``try``, ``except`` and ``finally`` , ``with``, ``assert``
and ``class``.
