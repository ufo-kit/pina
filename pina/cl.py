class ExecutionEnvironment(object):
    def __init__(self):
        self.MAX_CONSTANT_ARGS = 2
        self.MAX_CONSTANT_SIZE = 64 * 1024
        self.opt_level = 2


class BufferSpec(object):

    READ_ONLY = 0
    WRITE_ONLY = 1
    READ_WRITE = 2

    def __init__(self, name):
        self.name = name
        self.qualifier = None
        self.size = None
        self.shape = None
        self.access = self.READ_ONLY
