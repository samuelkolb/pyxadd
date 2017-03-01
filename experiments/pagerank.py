import numpy
import time
from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.matrix.matrix import Matrix


def to_stochastic(matrix_a):
    """
    :param Matrix matrix_a:
    :return:
    """
    projected = matrix_a.project(False)

    def indicate(terminal, diagram):
        return diagram.pool.terminal(1.0 / matrix_a.height if terminal.expression == 0 else 0)

    dangling = projected.transform_leaves(indicate)
    dangling = Matrix(dangling.diagram, matrix_a.row_variables, matrix_a.column_variables)

    matrix_a = matrix_a + make_constant(matrix_a, 1).element_product(dangling)

    def normalize(terminal, diagram):
        return diagram.pool.terminal(1.0 / terminal.expression if terminal.expression != 0 else 1.0)

    inverted = projected.transform_leaves(normalize)
    return matrix_a.element_product(inverted)


def make_constant(matrix, constant, build=None):
    if build is None:
        build = Builder(matrix.diagram.pool)
    ones = build.exp(constant)  # ones = build.exp(1.0 / matrix.height)
    for name, lb, ub in matrix.row_variables:
        ones = ones * build.limit(name, lb, ub)
    for name, lb, ub in matrix.column_variables:
        ones = ones * build.limit(name, lb, ub)
    return Matrix(ones, matrix.row_variables, matrix.column_variables)


def dampen(matrix_a, damping_factor, reduce_result=True):
    """
    :param Matrix matrix_a: The matrix to dampen
    :param float damping_factor: The dampening factor
    :param reduce_result: True if the dampened matrix should be reduced (default: True)
    :return Matrix: The dampened matrix
    """

    if damping_factor < 0 or damping_factor > 1:
        raise RuntimeError("Damping factor has to be in the interval [0, 1].")

    if damping_factor < 1:
        updated_a = damping_factor * matrix_a + (1 - damping_factor) * make_constant(matrix_a, 1.0 / matrix_a.height)
        if reduce_result:
            assert isinstance(updated_a, Matrix)
            updated_a = updated_a.reduce()
        return updated_a
    else:
        return matrix_a


def power_iteration(matrix_a, variables=None, initial_vector=None, iterations=100, delta=10 ** -3, norm=2):
    """
    Computes the pagerank of a matrix.
    By convention, the variables in the matrix are prefixed with r_ if they are used as row variables and c_ if they are
    used as column variables.
    :param Matrix matrix_a: The square, stochastic adjacency matrix
    :param variables: The variables of the matrix (if None it is inferred from the matrix)
    :param damping_factor: The damping factor used to prevent problems due to dangling or disconnected nodes
    :param Matrix initial_vector: The initial column vector (if None it is initialized with 1/N, where N is the size of the matrix)
    :param iterations: The maximal number of iterations before the algorithm stops
    :param delta: The threshold when change is considered negligible
    :param norm: The norm, default^ is 2 (L^2 norm), 1 is also allowed (L^1 norm)
    :return: The page rank vector
    """

    build = Builder(matrix_a.diagram.pool)

    if variables is None:
        raise RuntimeError("Variables computation not yet implemented.")
    names = [t[0] for t in variables]

    if initial_vector is None:
        initial_diagram = build.exp(1.0 / matrix_a.height)
        for name, lb, ub in variables:
            initial_diagram = initial_diagram * build.limit("r_" + name, lb, ub)
        row_vars = [("r_" + name, lb, ub) for name, lb, ub in variables]
        initial_vector = Matrix(initial_diagram, row_vars, [])

    initial_vector = initial_vector.rename({"r_" + var: var for var in names})

    previous_vector = None
    new_vector = initial_vector

    matrix_a = matrix_a.rename({"c_" + var: var for var in names})
    # matrix_a.export("speed/matrix")

    for i in range(iterations):
        # Check for convergence
        if previous_vector is not None:
            # Calculate difference vector
            difference_vector = new_vector - previous_vector

            # Compare norm of difference with given delta
            if difference_vector.norm(norm) < delta:
                return new_vector, i

        # Save previous vector
        previous_vector = new_vector

        # Compute next iteration
        new_vector = matrix_a * new_vector
        # new_vector.export("speed/vector{}".format(i))

        # Rename column variables to row variables
        new_vector = new_vector.rename({"r_" + var: var for var in names}).reduce()

    return new_vector, iterations


def pagerank(matrix_a, damping_factor, variables=None, initial_vector=None, iterations=100, delta=10 ** -3, norm=2):
    stochastic_a = to_stochastic(matrix_a)
    dampened_a = dampen(stochastic_a, damping_factor)
    return power_iteration(dampened_a, variables, initial_vector=initial_vector, iterations=iterations, delta=delta,
                           norm=norm)
