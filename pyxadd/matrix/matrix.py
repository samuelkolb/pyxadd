from __future__ import print_function

from pyxadd.leaf_transform import transform_leaves
from pyxadd.matrix_vector import sum_out
from pyxadd.rename import rename
from pyxadd.reduce import LinearReduction, SimpleBoundReducer


class Matrix(object):
    def __init__(self, diagram, row_vars, col_vars, height=None, width=None, auto_reduce=False, is_simple=None):
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
        self._is_simple = is_simple

    @property
    def row_variables(self):
        return self._row_vars

    @property
    def column_variables(self):
        return self._col_vars

    @property
    def diagram(self):
        return self._diagram

    @property
    def width(self):
        if self._width is None:
            self._width = 1
            for _, lb, ub in self._col_vars:
                self._width *= ub - lb + 1
        return self._width

    @property
    def height(self):
        if self._height is None:
            self._height = 1
            for _, lb, ub in self._row_vars:
                self._height *= ub - lb + 1
        return self._height

    def is_simple(self):
        from pyxadd.reduce import is_simple
        if self._is_simple is None:
            if len(self._row_vars + self._col_vars) <= 1:
                self._is_simple = True
            else:
                self._is_simple = is_simple(self.diagram)
        return self._is_simple

    def _matrix(self, diagram, row_vars, col_vars, matrices, height=None, width=None):
        """
        Constructs a matrix to be returned
        :type diagram: pyxadd.diagram.Diagram
        :type matrices: Matrix[]
        :rtype: Matrix
        """
        auto_reduce = all(matrix._auto_reduce for matrix in matrices)
        if all((matrix._is_simple is not None) for matrix in matrices):
            is_simple = all(matrix._is_simple for matrix in matrices)
        else:
            is_simple = None
        return Matrix(diagram, row_vars, col_vars, auto_reduce=auto_reduce, is_simple=is_simple, height=height,
                      width=width)

    def __add__(self, other):
        assert isinstance(other, Matrix)
        if self._row_vars != other._row_vars:
            raise RuntimeError("Mismatch between row variables: {} vs {}".format(self._row_vars, other._row_vars))
        if self._col_vars != other._col_vars:
            raise RuntimeError("Mismatch between row variables: {} vs {}".format(self._col_vars, other._col_vars))

        diagram = self.diagram + other.diagram
        diagram = self._optional_reduce(diagram)
        return self._matrix(diagram, self._row_vars, self._col_vars, [self, other], self.height, self.width)

    def __sub__(self, other):
        assert isinstance(other, Matrix)
        if self._row_vars != other._row_vars:
            raise RuntimeError("Mismatch between row variables {} and {}"
                               .format(self._row_vars, other._row_vars))
        assert self._col_vars == other._col_vars

        diagram = self.diagram - other.diagram
        diagram = self._optional_reduce(diagram)
        return self._matrix(diagram, self._row_vars, self._col_vars, [self, other], self.height, self.width)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            # Multiply with matrix
            if self._col_vars != other._row_vars:
                raise RuntimeError("Mismatch between column variables {} and row variables {}"
                                   .format(self._col_vars, other._row_vars))

            from pyxadd.matrix_vector import matrix_multiply
            pool = self.diagram.pool
            variables = [t[0] for t in self._col_vars]
            diagram = pool.diagram(matrix_multiply(pool, self.diagram.root_id, other.diagram.root_id, variables))
            diagram = self._optional_reduce(diagram)
            return self._matrix(diagram, self._row_vars, other._col_vars, [self, other], self.height, other.width)
        else:
            # Multiply with constant expression
            diagram = self.diagram * self.diagram.pool.diagram(self.diagram.pool.terminal(other))
            return self._matrix(diagram, self._row_vars, self._col_vars, [self], self.height, self.width)

    def __rmul__(self, other):
        return self * other

    def combine_variables(self, vars1, vars2):
        # SPEEDUP Use set instead of list for more variables
        result = list(vars1)
        for var in vars2:
            if var not in result:
                result.append(var)
        return result

    def element_product(self, other):
        if isinstance(other, Matrix):
            diagram = self.diagram * other.diagram
            # TODO check row / column variables
            row_vars = self.combine_variables(self._row_vars, other._row_vars)
            col_vars = self.combine_variables(self._col_vars, other._col_vars)
            return self._matrix(diagram, row_vars, col_vars, [self, other])
        else:
            raise RuntimeError("{} is not a matrix".format(other))

    def norm(self, l_norm=2):
        from pyxadd.norm import norm
        return norm([t[0] for t in self._row_vars] + [t[0] for t in self._col_vars], self.diagram, l_norm)

    def reduce(self, reducer=None):
        diagram = self._reduce_diagram(self.diagram, reducer)
        return self._matrix(diagram, self._row_vars, self._col_vars, [self], self.height, self.width)

    def evaluate(self, assignment):
        return self.diagram.evaluate(assignment)

    def rename(self, translation):
        diagram = self.diagram.pool.diagram(rename(self.diagram, translation))
        row_vars = [(translation[v] if v in translation else v, lb, ub) for v, lb, ub in self._row_vars]
        col_vars = [(translation[v] if v in translation else v, lb, ub) for v, lb, ub in self._col_vars]
        return self._matrix(diagram, row_vars, col_vars, [self], self.height, self.width)

    def transpose(self):
        return self._matrix(self.diagram, self._col_vars, self._row_vars, [self], self.width, self.height)

    def print_ground(self, row_limit=None, column_limit=None):
        results = []

        grounded = self.to_ground(row_limit=row_limit, column_limit=column_limit)

        if len(self._col_vars) > 0:
            header = []
            if len(self._row_vars) > 0:
                header.append("")
            labels = []
            label_index = 0
            for a in assignments(self._col_vars):
                if column_limit is not None and label_index >= column_limit:
                    break
                labels.append(str(a))
                label_index += 1
            header += labels
            results.append(header)

        row_index = 0
        for row_assignment in assignments(self._row_vars):
            if row_limit is not None and row_index >= row_limit:
                break
            row = []
            if len(self._row_vars) > 0:
                row.append(str(row_assignment))
            row_values = []
            for entry in grounded[row_index]:
                row_values.append(str(entry))
            row += row_values
            results.append(row)
            row_index += 1

        padding = max(max(len(entry) for entry in row) for row in results)
        results = [[entry.rjust(padding) for entry in row] for row in results]

        for row in results:
            print(*row, sep="  ")

    def to_ground(self, row_limit=None, column_limit=None):
        matrix = []
        row_index = 0
        for row_assignment in assignments(self._row_vars):
            if row_limit is not None and row_index >= row_limit:
                break
            row = []
            column_index = 0
            for a in assignments(self._col_vars, fixed=row_assignment):
                if column_limit is not None and column_index >= column_limit:
                    break
                row.append(self.evaluate(a))
                column_index += 1
            matrix.append(row)
            row_index += 1
        return matrix

    def export(self, name):
        from pyxadd.view import export
        if not name.endswith(".dot"):
            name += ".dot"
        export(self.diagram, name)

    def project(self, on_rows):
        """
        :param bool on_rows: If true, project on rows, otherwise on columns
        :return Matrix: A matrix corresponding to the projection
        """
        to_sum_out = self.column_variables if on_rows else self.row_variables
        projected_id = sum_out(self.diagram.pool, self.diagram.root_id, [v[0] for v in to_sum_out])
        new_row_vars = self.row_variables if on_rows else []
        new_col_vars = self.column_variables if not on_rows else []
        projected_diagram = self.diagram.pool.diagram(projected_id)
        return self._matrix(projected_diagram, new_row_vars, new_col_vars, [self])

    def transform_leaves(self, f):
        transformed_id = transform_leaves(f, self.diagram)
        diagram = self.diagram.pool.diagram(transformed_id)
        return self._matrix(diagram, self.row_variables, self.column_variables, [self], self.height, self.width)

    def _optional_reduce(self, diagram):
        if self._auto_reduce is not False:
            return self._reduce_diagram(diagram)
        else:
            return diagram

    def _reduce_diagram(self, diagram, reducer=None):
        if reducer is None:
            reducer = self._reducer if not self.is_simple() else self._simple_reducer

        variables = [t[0] for t in self._row_vars + self._col_vars]
        # print(variables)
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
