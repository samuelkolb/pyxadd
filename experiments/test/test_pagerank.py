from __future__ import print_function

import unittest

import numpy
from experiments import pagerank
from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.matrix.matrix import Matrix


def simple(matrix, alpha, delta, iterations):
    n = matrix.shape[0]

    print("--- Sparse ---")

    # To sparse
    from scipy.sparse import csr_matrix
    sparse_matrix = csr_matrix(matrix, dtype=numpy.float)
    print(sparse_matrix.todense())

    # To stochastic
    column_sums = numpy.array(sparse_matrix.sum(0))[0, :]  # Used to be sparse_matrix.sum(1), but I sum columns.
    ri, ci = sparse_matrix.nonzero()
    sparse_matrix.data /= column_sums[ci]

    print("Stochastic")
    print(sparse_matrix.todense())

    # Bool array of sink states
    sink = (column_sums == 0) / float(n)
    sink_alpha = sink * alpha

    print("Sink")
    print(sink)

    alpha_matrix = sparse_matrix * alpha
    ones = numpy.ones(n) / float(n)
    print("Ones")
    print(ones)

    # Account for teleportation
    teleportation = numpy.ones(n) / float(n)
    teleportation_alpha = (1 - alpha) * teleportation

    # Compute pagerank r until we converge
    ro, r = numpy.zeros(n), numpy.ones(n) / float(n)
    print("R initial:")
    print(r)
    iteration = 0
    while iteration == 0 or (numpy.linalg.norm(r - ro) > delta and iteration < iterations):
        ro = r.copy()
        print(ro, sparse_matrix.todense(), ro * sparse_matrix)
        r = alpha * sparse_matrix * ro + (alpha * ro * sink + (1 - alpha)) * ones
        print("R iteration: " + str(iteration + 1))
        print(r)
        iteration += 1

    # return normalized pagerank
    return r / sum(r), iteration


def pagerank_ground(matrix_a, damping_factor, iterations, delta, initial_vector=None):
    n = matrix_a.shape[0]
    matrix_a = numpy.matrix(matrix_a, dtype=float)

    projected = numpy.zeros([n, 1])
    for i in range(n):
        projected[i] = sum(matrix_a[:, i])

    for i in range(n):
        for j in range(n):
            if projected[j] == 0:
                matrix_a[i, j] = 1.0 / n
            else:
                matrix_a[i, j] = matrix_a[i, j] / projected[j]

    if initial_vector is None:
        initial_vector = numpy.matrix([[1.0 / n for _ in range(n)]]).T

    if damping_factor < 0 or damping_factor > 1:
        raise RuntimeError("Damping factor has to be in the interval [0, 1].")

    previous_vector = None
    new_vector = initial_vector

    if damping_factor < 1:
        ones = numpy.matrix([[1.0 / n for row in range(n)] for col in range(n)])
        matrix_a = damping_factor * matrix_a + (1 - damping_factor) * ones

    for i in range(iterations):
        # Check for convergence
        if previous_vector is not None:
            # Calculate difference vector
            difference_vector = new_vector - previous_vector

            # Compare norm of difference with given delta
            if numpy.linalg.norm(difference_vector) < delta:
                return new_vector, i

        # Save previous vector and compute next iteration
        previous_vector = new_vector
        new_vector = matrix_a * new_vector

    return new_vector, iterations


def pagerank_scipy(G, alpha=0.85, personalization=None,
                   max_iter=100, tol=1.0e-6, weight='weight',
                   dangling=None):
    import warnings
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    import scipy.sparse
    import scipy.linalg

    G = G.T
    N = len(G)
    if N == 0:
        return {}

    nodelist = list(range(N))
    M = scipy.sparse.coo_matrix(G, dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # initial vector
    x = scipy.repeat(1.0 / N, N)

    # Personalization vector
    if personalization is None:
        p = scipy.repeat(1.0 / N, N)
    else:
        p = scipy.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        p = p / p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = scipy.array([dangling.get(n, 0) for n in nodelist],
                                       dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = scipy.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
            (1 - alpha) * p
        # check convergence, l1 norm
        # err = scipy.absolute(x - xlast).sum()
        err = scipy.linalg.norm(x - xlast)
        if err < tol:
            return map(float, x), i + 1
    raise RuntimeError("Failed to converge")


def pagerank_sparse(matrix, alpha, delta, iterations):
    n = matrix.shape[0]
    print("SPARSE")
    # transform G into markov matrix M
    from scipy.sparse import csc_matrix
    M = csc_matrix(matrix, dtype=numpy.float)
    rsums = numpy.array(M.sum(0))[0, :]  # Used to be M.sum(1), but I sum columns...?
    ri, ci = M.nonzero()
    M.data /= rsums[ci]
    print(M.todense())

    # bool array of sink states
    sink = rsums == 0

    # Compute pagerank r until we converge
    ro, r = numpy.zeros(n), numpy.ones(n) / float(n)
    iteration = 0
    print(r)
    while numpy.linalg.norm(r - ro) > delta and iteration < iterations:
        ro = r.copy()
        # calculate each pagerank at a time
        for i in xrange(0, n):
            # inlinks of state i
            Ii = numpy.array(M[i, :].todense())[0, :]
            # account for sink states
            Si = sink / float(n)
            # account for teleportation to state i
            Ti = numpy.ones(n) / float(n)

            r[i] = ro.dot(Ii * alpha + Si * alpha + Ti * (1 - alpha))
        iteration += 1
        print(r)

    # return normalized pagerank
    return r / sum(r), iteration


class TestPagerank(unittest.TestCase):
    def setUp(self):
        self.print_time = False
        self.delta = 10 ** -3
        self.damping_factor = 0.85

    @staticmethod
    def build_example1():
        # http://www.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
        pool = Pool()
        build = Builder(pool)
        variables = [("i", 1, 4)]
        for var in variables:
            name = var[0]
            build.ints("r_{}".format(name), "c_{}".format(name), name)
        limits = build.limit("r_i", 1, 4) & build.limit("c_i", 1, 4)
        column1 = (build.limit("c_i", 1, 1) & build.limit("r_i", 2, 4)) * build.exp("1/3")
        column2 = (build.limit("c_i", 2, 2) & build.limit("r_i", 3, 4)) * build.exp("1/2")
        column3 = (build.limit("c_i", 3, 3) & build.limit("r_i", 1, 1)) * build.exp("1")
        column4 = (build.limit("c_i", 4, 4) & (build.limit("r_i", 1, 1) | build.limit("r_i", 3, 3))) * build.exp("1/2")
        diagram = limits * (column1 + column2 + column3 + column4)
        return diagram, variables

    @staticmethod
    def build_example2():
        # http://www.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
        # Dangling
        pool = Pool()
        build = Builder(pool)
        variables = [("i", 1, 3)]
        for var in variables:
            name = var[0]
            build.ints("r_{}".format(name), "c_{}".format(name), name)
        limits = build.limit("r_i", 1, 3) & build.limit("c_i", 1, 3)
        diagram = limits * (build.limit("r_i", 3, 3) & build.limit("c_i", 1, 2))
        return diagram, variables

    def test_example1(self):
        diagram, variables = self.build_example1()
        self._test_compare(diagram, variables, 100, self.damping_factor)

    def test_example2(self):
        diagram, variables = self.build_example2()
        self._test_compare(diagram, variables, 100, self.damping_factor)

    def test_block_example(self):
        from experiments import link_prediction
        matrix, variables = link_prediction.build_block_diagram()
        self._test_compare(matrix.diagram, variables, 100, self.damping_factor)

    def _test_compare(self, diagram, variables, iterations, damping_factor):
        row_variables = [("r_" + name, lb, ub) for name, lb, ub in variables]
        column_variables = [("c_" + name, lb, ub) for name, lb, ub in variables]
        matrix = Matrix(diagram, row_variables, column_variables).reduce()
        ground_matrix = numpy.matrix(matrix.to_ground())
        print(ground_matrix)
        result_xadd, iterations_xadd = pagerank.pagerank(matrix,
                                                         damping_factor=damping_factor,
                                                         variables=variables,
                                                         iterations=iterations,
                                                         delta=self.delta)

        result_ground, iterations_ground = pagerank_ground(ground_matrix,
                                                           damping_factor=damping_factor,
                                                           iterations=iterations,
                                                           delta=self.delta)

        grounded_xadd = numpy.matrix(result_xadd.to_ground())
        sparse1 = pagerank_scipy(ground_matrix, alpha=damping_factor, max_iter=iterations, tol=self.delta)

        sparse2, _ = pagerank_sparse(ground_matrix, alpha=damping_factor, delta=self.delta, iterations=iterations)

        for i in range(len(result_ground)):
            self.assertAlmostEquals(grounded_xadd[i], result_ground[i], delta=10 ** -3)
            self.assertAlmostEquals(grounded_xadd[i], sparse1[i], delta=10 ** -3)
        self.assertAlmostEquals(grounded_xadd[i], sparse1[i], delta=10 ** -3)
        difference = grounded_xadd - result_ground
        self.assertTrue(numpy.linalg.norm(difference) < 2 * self.delta)
        self.assertEqual(iterations_xadd, iterations_ground)

    def test_consistency(self):
        diagram, variables = self.build_example1()
        self._test_compare_methods(diagram, variables, 20, 1)

    def _test_compare_methods(self, diagram, variables, iterations, damping_factor):
        row_variables = [("r_" + name, lb, ub) for name, lb, ub in variables]
        column_variables = [("c_" + name, lb, ub) for name, lb, ub in variables]
        matrix = Matrix(diagram, row_variables, column_variables).reduce()
        ground_matrix = numpy.matrix(matrix.to_ground())

        print(ground_matrix)

        result_xadd, iterations_xadd = pagerank.pagerank(matrix,
                                                         damping_factor=damping_factor,
                                                         variables=variables,
                                                         iterations=iterations,
                                                         delta=self.delta)

        result_ground, iterations_ground = pagerank_ground(ground_matrix,
                                                           damping_factor=damping_factor,
                                                           iterations=iterations,
                                                           delta=self.delta)

        grounded_xadd = numpy.matrix(result_xadd.to_ground())
        sparse1, _ = pagerank_scipy(ground_matrix, alpha=damping_factor, max_iter=iterations, tol=self.delta)

        sparse2, _ = pagerank_sparse(ground_matrix, alpha=damping_factor, delta=self.delta, iterations=iterations)

        sparse3, _ = simple(ground_matrix, damping_factor, self.delta, iterations)

        print("XADD")
        print(grounded_xadd)
        print()
        print("Numpy (ground)")
        print(result_ground)
        print()
        print("Sparse (networkx)")
        print(sparse1)
        print()
        print("Sparse (online)")
        print(sparse2)
        print()
        print("Sparse (simple)")
        print(sparse3)
