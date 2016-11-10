class Matrix:

    def __init__(self, width, height, diagram):
        self._width = width
        self._height = height
        self._diagram = diagram

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def diagram(self):
        return self._diagram


def assignments(variables, fixed=None):
    if fixed is None:
        fixed = dict()

    if len(variables) == 0:
        yield fixed
    else:
        name, lb, ub = variables[0]
        new_variables = list(variables)
        del new_variables[0]
        for v in range(lb, ub + 1):
            new_fixed = dict(fixed)
            new_fixed[name] = v
            for a in assignments(new_variables, new_fixed):
                yield a
