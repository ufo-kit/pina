_TYPE_MAP = {
    "<type 'bool'>": 'bool',
    "<type 'float'>": 'double',
    "<type 'int'>": 'int',
    "<type 'str'>": 'char',
    "<type 'numpy.bool'>": 'bool',
    "<type 'numpy.int'>": 'int',
    "<type 'numpy.int8'>": 'char',
    "<type 'numpy.uint8'>": 'unsigned char',
    "<type 'numpy.float'>": 'double',
    "<type 'numpy.float16'>": 'half',
    "<type 'numpy.float32'>": 'float',
    "<type 'numpy.float64'>": 'double',
}


class AddressSpaceQualifier(object):
    def __init__(self, python_type):
        # Using a string representations frees us from importing (and
        # depending) on large modules like NumPy.
        type_repr = repr(python_type)
        self.type_name = _TYPE_MAP.get(type_repr, 'float')


class NoQualifier(AddressSpaceQualifier):
    def __init__(self, python_type):
        super(NoQualifier, self).__init__(python_type)
        self.cl_keyword = ''


class Global(AddressSpaceQualifier):
    def __init__(self, python_type):
        super(Global, self).__init__(python_type)
        self.cl_keyword = '__global'


class Constant(AddressSpaceQualifier):
    def __init__(self, python_type):
        super(Constant, self).__init__(python_type)
        self.cl_keyword = '__constant'


class Local(AddressSpaceQualifier):
    def __init__(self, python_type):
        super(Local, self).__init__(python_type)
        self.cl_keyword = '__local'
