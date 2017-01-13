import numpy
import time
from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.matrix.matrix import Matrix


def dampen(matrix_a, damping_factor, reduce_result=True):
    """
    :param Matrix matrix_a: The matrix to dampen
    :param float damping_factor: The dampening factor
    :param reduce_result: True if the dampened matrix should be reduced (default: True)
    :return Matrix: The dampened matrix
    """
    build = Builder(matrix_a.diagram.pool)

    if damping_factor < 0 or damping_factor > 1:
        raise RuntimeError("Damping factor has to be in the interval [0, 1].")

    if damping_factor < 1:
        ones = build.exp(1.0 / matrix_a.height)
        for name, lb, ub in matrix_a.row_variables:
            ones = ones * build.limit(name, lb, ub)
        for name, lb, ub in matrix_a.column_variables:
            ones = ones * build.limit(name, lb, ub)
        matrix_ones = Matrix(ones, matrix_a.row_variables, matrix_a.column_variables)
        updated_a = damping_factor * matrix_a + (1 - damping_factor) * matrix_ones
        if reduce_result:
            assert isinstance(updated_a, Matrix)
            updated_a = updated_a.reduce()
        return updated_a
    else:
        return matrix_a


def page_rank(matrix_a, variables=None, initial_vector=None, iterations=100, delta=10**-3):
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

    for i in range(iterations):
        # Check for convergence
        if previous_vector is not None:
            # Calculate difference vector
            difference_vector = new_vector - previous_vector

            # Compare norm of difference with given delta
            if difference_vector.norm() < delta:
                return new_vector, i

        # Save previous vector
        previous_vector = new_vector

        # Compute next iteration
        new_vector = matrix_a * new_vector

        # Rename column variables to row variables
        new_vector = new_vector.rename({"r_" + var: var for var in names}).reduce()

    return new_vector, iterations
