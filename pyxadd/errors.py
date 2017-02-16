class UnexpectedTypeError(RuntimeError):
    def __init__(self, kind, target):
        super("Unexpected {} {} of type {}".format(kind, target, type(target)))
        self.kind = kind
        self.target = target
