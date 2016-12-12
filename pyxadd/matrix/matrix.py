from __future__ import print_function

from pyxadd.rename import rename
from pyxadd.reduce import LinearReduction, SimpleBoundReducer


class Matrix(object):
    def __init__(self, diagram, row_vars, col_vars, height=None, width=None, auto_reduce=False):
        from pyxadd.diagram import Diagram
        assert isinstance(diagram, Diagram)
        self._diagram = diagram
        self._row_vars = row_vars
        self._col_vars = col_vars
        self._height = height
        self._width = width
        self._auto_reduce = auto_reduce
        self._reducer = LinearReduction(self.diagram.pool)
        self._simple_reducer = SimpleBoundReducer(self.diagram.pool)

    @property
    def diagram(self):
        return self._diagram

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __add__(self, other):
        assert isinstance(other, Matrix)
        assert self._row_vars == other._row_vars
        assert self._col_vars == other._col_vars

        diagram = self.diagram + other.diagram
        diagram = self._optional_reduce(diagram)
        return Matrix(diagram, self._row_vars, self._col_vars, height=self.height, width=self.width,
                      auto_reduce=(self._auto_reduce and other._auto_reduce))

    def __sub__(self, other):
        assert isinstance(other, Matrix)
        if self._row_vars != other._row_vars:
            raise RuntimeError("Mismatch between row variables {} and {}"
                               .format(self._row_vars, other._row_vars))
        assert self._col_vars == other._col_vars

        diagram = self.diagram - other.diagram
        diagram = self._optional_reduce(diagram)
        return Matrix(diagram, self._row_vars, self._col_vars, height=self.height, width=self.width,
                      auto_reduce=(self._auto_reduce and other._auto_reduce))

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self._col_vars != other._row_vars:
                raise RuntimeError("Mismatch between column variables {} and row variables {}"
                                   .format(self._col_vars, other._row_vars))

            from pyxadd.matrix_vector import matrix_multiply
            pool = self.diagram.pool
            variables = [t[0] for t in self._col_vars]
            diagram = pool.diagram(matrix_multiply(pool, self.diagram.root_id, other.diagram.root_id, variables))
            diagram = self._optional_reduce(diagram)
            return Matrix(diagram, self._row_vars, other._col_vars, height=self.height, width=other.width,
                          auto_reduce=(self._auto_reduce and other._auto_reduce))
        else:
            diagram = self.diagram * self.diagram.pool.diagram(self.diagram.pool.terminal(other))
            return Matrix(diagram, self._row_vars, self._col_vars, height=self.height, width=self.width,
                          auto_reduce=self._auto_reduce)

    def __rmul__(self, other):
        return self * other

    def element_product(self, other):
        if isinstance(other, Matrix):
            diagram = self.diagram * other.diagram
            # TODO check row / column variables
            return Matrix(diagram, self._row_vars + other._row_vars, self._col_vars + other._row_vars,
                          auto_reduce = (self._auto_reduce and other._auto_reduce))
        else:
            raise RuntimeError("{} is not a matrix".format(other))

    def norm(self):
        from pyxadd.norm import norm
        return norm([t[0] for t in self._row_vars] + [t[0] for t in self._col_vars], self.diagram)

    def reduce(self, reducer=None):
        diagram = self._reduce_diagram(self.diagram, reducer)
        return Matrix(diagram, self._row_vars, self._col_vars, height=self.height, width=self.width,
                      auto_reduce=self._auto_reduce)

    def evaluate(self, assignment):
        return self.diagram.evaluate(assignment)

    def rename(self, translation):
        diagram = self.diagram.pool.diagram(rename(self.diagram, translation))
        row_vars = [(translation[v] if v in translation else v, lb, ub) for v, lb, ub in self._row_vars]
        col_vars = [(translation[v] if v in translation else v, lb, ub) for v, lb, ub in self._col_vars]
        return Matrix(diagram, row_vars, col_vars, height=self.height, width=self.width,
                      auto_reduce=self._auto_reduce)

    def transpose(self):
        return Matrix(self.diagram, self._col_vars, self._row_vars, auto_reduce=self._auto_reduce)

    def print_ground(self):
        results = []

        if len(self._col_vars) > 0:
            header = []
            if len(self._row_vars) > 0:
                header.append("")
            header += [str(a) for a in assignments(self._col_vars)]
            results.append(header)

        for row_assignment in assignments(self._row_vars):
            row = []
            if len(self._row_vars) > 0:
                row.append(str(row_assignment))
            row += [str(self.evaluate(a)) for a in assignments(self._col_vars, fixed=row_assignment)]
            results.append(row)

        padding = max(max(len(entry) for entry in row) for row in results)
        results = [[entry.rjust(padding) for entry in row] for row in results]

        for row in results:
            print(*row, sep="  ")

    def export(self, name):
        from pyxadd.view import export
        if not name.endswith(".dot"):
            name += ".dot"
        export(self.diagram, name)

    def _optional_reduce(self, diagram):
        if self._auto_reduce is not False:
            return self._reduce_diagram(diagram)
        else:
            return diagram

    def _reduce_diagram(self, diagram, reducer=None):
        if reducer is None:
            reducer = self._reducer if len(self._row_vars + self._col_vars) > 1 else self._simple_reducer

        variables = [t[0] for t in self._row_vars + self._col_vars]
        print(variables)
        return diagram.pool.diagram(reducer.reduce(diagram.root_id, variables))


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
