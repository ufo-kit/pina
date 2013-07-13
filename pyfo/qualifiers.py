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

_TYPE_PRIORITY = {
    'bool': 1,
    'char': 2,
    'unsigned char': 2,
    'int': 3,
    'unsigned int': 3,
    'half': 4,
    'float': 5,
    'double': 6
}


def set_default_float_type(type_name):
    _TYPE_MAP["<type 'float'>"] = type_name


class AddressSpaceQualifier(object):
    def __init__(self, python_type):
        # Using a string representations frees us from importing (and
        # depending) on large modules like NumPy.
        self.type_repr = repr(python_type)

    @property
    def type_name(self):
        return _TYPE_MAP.get(self.type_repr, 'float')

    def priority(self):
        return _TYPE_PRIORITY[self.type_name]

    def __lt__(self, other):
        return self.priority() < other.priority()

    def __le__(self, other):
        return self.priority() <= other.priority()

    def __eq__(self, other):
        return self.priority() == other.priority()

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        return not self < other

    def __ge__(self, other):
        return not self <= other


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
