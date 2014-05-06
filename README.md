## Piña

Piña is a Python-to-OpenCL translator. And a fruit.


### Usage

Import the `static` decorator to turn any Python function to OpenCL kernel code:

```python
from pina import static

@static
def add(x, y):
    return 2.0 * cos(x) + sin(y)
```

Any return statement will generate an implicit write to the hidden `output`
kernel argument.

Executing the Python code requires a run-time system environment. This is based
on [PyOpenCL][] so make sure to install it first:

```python
from pina.ext.pycl import Runtime

r = Runtime()

@r.jit
def add(x, y):
    return 2.0 * cos(x) + sin(y)
```


### Indexing

Omitting square brackets reading and writing values will be local to the
executed kernel. By using square brackets, adjacent data can be accessed in a
relative way:

```python
@static
def foo(x):
    left = x[-1]
    right = x[+1]
    top_left = x[-1, -1]
    bottom_left = x[-1, +1]
```


### Type annotations

Static type annotations are written as simple decorator arguments. All integral
builtin types as well as all NumPy types are supported::

```python
@static(float, np.uint8)
def bar(x, y):
    pass
```

OpenCL address space qualifiers can be specified by using the appropriate
qualifier classes::

```python
from pina import Global, Constant

@static(Global(np.float32), Constant)
def qux(x, y):
    pass
```


### Unsupported Python constructs

Some Python features cannot be mapped reasonably onto OpenCL, e.g.  ``import``,
``del``, ``lambda``, ``try``, ``except`` and ``finally`` , ``with``, ``assert``
and ``class``.

[PyOpenCL]: http://mathema.tician.de/software/pyopencl/
