class Operation:
    @classmethod
    def apply(cls, diagram1, diagram2):
        raise NotImplementedError()


class Times(Operation):
    @classmethod
    def apply(cls, diagram1, diagram2):
        pass
