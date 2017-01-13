from __future__ import print_function

import unittest

import numpy
from experiments import pagerank
from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.matrix.matrix import Matrix


def pagerank_ground(matrix_a, damping_factor, iterations, delta, initial_vector=None):
    n = matrix_a.shape[0]
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


class TestPagerank(unittest.TestCase):
    def setUp(self):
        self.print_time = False
        self.delta = 10 ** -3
        self.damping_factor = 0.85

    @staticmethod
    def build_example1():
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

    def test_example1(self):
        diagram, variables = self.build_example1()
        self._test_compare(diagram, variables, 100, self.damping_factor)

    def _test_compare(self, diagram, variables, iterations, damping_factor):
        row_variables = [("r_" + name, lb, ub) for name, lb, ub in variables]
        column_variables = [("c_" + name, lb, ub) for name, lb, ub in variables]
        matrix = Matrix(diagram, row_variables, column_variables).reduce()
        ground_matrix = numpy.matrix(matrix.to_ground())
        matrix = pagerank.dampen(matrix, damping_factor=damping_factor)
        result_xadd, iterations_xadd = pagerank.page_rank(matrix, variables, iterations=iterations, delta=self.delta)

        result_ground, iterations_ground = pagerank_ground(ground_matrix, damping_factor=damping_factor,
                                                           iterations=iterations, delta=self.delta)
        difference = numpy.matrix(result_xadd.to_ground()) - result_ground
        self.assertTrue(numpy.linalg.norm(difference) < self.delta)
        self.assertEqual(iterations_xadd, iterations_ground)
