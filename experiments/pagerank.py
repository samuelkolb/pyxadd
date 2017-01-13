import numpy
import time
from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.matrix.matrix import Matrix


def page_rank(matrix_a, variables=None, damping_factor=0.85, initial_vector=None, iterations=100, delta=10**-3):
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

    if damping_factor < 0 or damping_factor > 1:
        raise RuntimeError("Damping factor has to be in the interval [0, 1].")

    initial_vector = initial_vector.rename({"r_" + var: var for var in names})

    previous_vector = None
    new_vector = initial_vector

    if damping_factor < 1:
        ones = build.exp(1.0 / matrix_a.height)
        for name, lb, ub in variables:
            ones = ones * build.limit("r_" + name, lb, ub) * build.limit("c_" + name, lb, ub)
        row_vars = [("r_" + name, lb, ub) for name, lb, ub in variables]
        col_vars = [("c_" + name, lb, ub) for name, lb, ub in variables]
        matrix_ones = Matrix(ones, row_vars, col_vars)
        updated_a = damping_factor * matrix_a + (1 - damping_factor) * matrix_ones
        matrix_a = updated_a.reduce()
        matrix_a.print_ground()

    print()
    matrix_a = matrix_a.rename({"c_" + var: var for var in names})

    for i in range(iterations):
        # Check for convergence
        if previous_vector is not None:
            # Calculate difference vector
            difference_vector = new_vector - previous_vector

            # Compare norm of difference with given delta
            if difference_vector.norm() < delta:
                print("Found solution after {} iterations".format(i))
                return new_vector

        print("Iteration: {}".format(i + 1))

        # Save previous vector
        previous_vector = new_vector

        # Compute next iteration
        new_vector = matrix_a * new_vector

        # Rename column variables to row variables
        new_vector = new_vector.rename({"r_" + var: var for var in names}).reduce()

        # Print for debugging
        new_vector.print_ground()

    print("No solution after {} iterations".format(iterations))
    return new_vector


def example1():
    pool = Pool()
    build = Builder(pool)
    build.ints("r_i", "c_i", "i")
    limits = build.limit("r_i", 1, 4) & build.limit("c_i", 1, 4)
    column1 = (build.limit("c_i", 1, 1) & build.limit("r_i", 2, 4)) * build.exp("1/3")
    column2 = (build.limit("c_i", 2, 2) & build.limit("r_i", 3, 4)) * build.exp("1/2")
    column3 = (build.limit("c_i", 3, 3) & build.limit("r_i", 1, 1)) * build.exp("1")
    column4 = (build.limit("c_i", 4, 4) & (build.limit("r_i", 1, 1) | build.limit("r_i", 3, 3))) * build.exp("1/2")
    diagram = limits * (column1 + column2 + column3 + column4)
    r = ("r_i", 1, 4)
    c = ("c_i", 1, 4)
    i = ("i", 1, 4)
    matrix = Matrix(diagram, [r], [c], height=4, width=4).reduce()
    # vector = Matrix(build.limit("r_i", 1, 4) * build.exp("1/4"), [r], [])

    matrix.print_ground()
    matrix.export("visual/pagerank/matrix.dot")
    # vector.print_ground()
    result = page_rank(matrix, variables=[i], iterations=100)
    result.export("visual/pagerank/result.dot")
    result.print_ground()


def numerical_example1():
    matrix = numpy.matrix([[0, 0, 1, 0.5], [1/3.0, 0, 0, 0], [1/3.0, 0.5, 0, 0.5], [1/3.0, 0.5, 0, 0]])
    new_vector = numpy.matrix([[0.25, 0.25, 0.25, 0.25]]).T
    previous = None
    for i in range(100):
        if previous is not None:
            difference_vector = new_vector - previous
            if numpy.linalg.norm(difference_vector) < 10**-3:
                print("Found solution after {} iterations".format(i))
                return new_vector
        print("Iteration: {}".format(i + 1))
        previous = new_vector
        new_vector = matrix * new_vector
        print(new_vector)


# start = time.time()
# example1()
# print(time.time() - start)


def example2():
    pool = Pool()
    build = Builder(pool)
    build.ints("r_i", "c_i", "i")
