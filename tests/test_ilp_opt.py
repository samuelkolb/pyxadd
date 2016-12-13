import unittest

from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.optimization import find_optimum
from pyxadd.view import export


def build_diagram1():
    pool = Pool()
    b = Builder(pool)
    b.ints("x")
    return b.ite(b.test("x", "<=", 3), b.ite(b.test("x", "<=", 2), b.exp(1), b.exp("2*x")), b.exp(1))


delta = 10 ** -6


class TestIlpOpt(unittest.TestCase):
    def test_diagram1(self):
        diagram = build_diagram1()
        value, assignment = find_optimum(diagram)
        self.assertAlmostEqual(6.0, value, delta=delta)
        self.assertEqual(1, len(assignment))
        self.assertAlmostEqual(3.0, assignment["x"], delta=delta)
