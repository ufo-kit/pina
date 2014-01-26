__version__ = '0.0.1-dev'

from .gen import kernel
from .misc import static, jit, invoked
from .qualifiers import Global, Constant, Local
from .qualifiers import set_default_float_type
