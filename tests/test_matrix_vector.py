from __future__ import print_function

import unittest

from pyxadd.build import Builder
from pyxadd.diagram import Diagram, Pool
from pyxadd.matrix_vector import SummationWalker, matrix_multiply
from pyxadd.partial import PartialWalker
from pyxadd.reduce import SmtReduce, LinearReduction
from pyxadd.test import LinearTest
from pyxadd.view import export


class TestMatrixVector(unittest.TestCase):
    def setUp(self):
        self.diagram = TestMatrixVector.construct_diagram()

    @staticmethod
    def construct_diagram():
        pool = Pool()
        pool.int_var("x", "y")

        b = Builder(pool)
        bounds = b.test("x", ">=", 0) & b.test("x", "<=", 8) & b.test("y", ">=", 1) & b.test("y", "<=", 10)
        return bounds * b.ite(b.test("x", ">=", "y"), b.terminal("2*x + 3*y"), b.terminal("3*x + 2*y"))

    # FIXME In case I forget: Introduce ordering on integer comparisons

    def test_summation_one_var(self):
        pool = Pool()
        pool.add_var("x", "int")
        pool.add_var("y", "int")
        b = Builder(pool)
        bounds = b.test("x", ">=", 0) & b.test("x", "<=", 10)
        d = b.ite(bounds, b.terminal("x"), b.terminal(0))
        d_const = Diagram(pool, SummationWalker(d, "x").walk())
        self.assertEqual(55, d_const.evaluate({}))

    def test_summation_two_var(self):
        pool = Pool()
        pool.add_var("x", "int")
        pool.add_var("y", "int")
        b = Builder(pool)
        bounds = b.test("x", ">=", 0) & b.test("x", "<=", 10)
        bounds &= b.test("y", ">=", 0) & b.test("y", "<=", 1)
        d = b.ite(bounds, b.terminal("x"), b.terminal(0))
        d_const = Diagram(pool, SummationWalker(d, "x").walk())
        for y in range(2):
            self.assertEqual(55, d_const.evaluate({"y": y}))

    def test_summation_two_var_test(self):
        pool = Pool()
        pool.add_var("x", "int")
        pool.add_var("y", "int")
        b = Builder(pool)
        bounds = b.test("x", ">=", 0) & b.test("x", "<=", 1)
        bounds &= b.test("y", ">=", 1) & b.test("y", "<=", 3)
        two = b.test("x", ">=", "y")
        d = b.ite(bounds, b.ite(two, b.terminal("x"), b.terminal("10")), b.terminal(0))

        summed = Diagram(pool, SummationWalker(d, "x").walk())
        d_const = summed.reduce(["y"])
        export(d_const, "summed_reduced.dot")
        for y in range(-20, 20):
            s = 0
            for x in range(-20, 20):
                s += d.evaluate({"x": x, "y": y})
            self.assertEqual(s, d_const.evaluate({"y": y}))

    def test_mixed_symbolic(self):
        diagram_y = Diagram(self.diagram.pool, SummationWalker(self.diagram, "x").walk())
        diagram_y = Diagram(diagram_y.pool, LinearReduction(diagram_y.pool).reduce(diagram_y.root_node.node_id, ["y"]))

        for y in range(0, 12):
            row_result = 0
            for x in range(0, 12):
                row_result += self.diagram.evaluate({"x": x, "y": y})
            self.assertEqual(diagram_y.evaluate({"y": y}), row_result)

    def test_partial(self):
        partial = PartialWalker(self.diagram, {"y": 2}).walk()
        for x in range(-10, 10):
            if x < 0 or x > 8:
                self.assertEqual(0, partial.evaluate({"x": x}))
            elif x > 2:
                self.assertEqual(2 * x + 6, partial.evaluate({"x": x}))
            else:
                self.assertEqual(3 * x + 4, partial.evaluate({"x": x}))

    def test_multiplication(self):
        pool = Pool()
        pool.int_var("x1", "x2")
        x_two = Diagram(pool, pool.terminal("x2"))
        two = Diagram(pool, pool.terminal("2"))
        three = Diagram(pool, pool.terminal("3"))
        four = Diagram(pool, pool.terminal("4"))

        test11 = Diagram(pool, pool.bool_test(LinearTest("x1", ">=")))
        test12 = Diagram(pool, pool.bool_test(LinearTest("x1 - 1", "<=")))
        test13 = Diagram(pool, pool.bool_test(LinearTest("x1 - 3", ">")))

        test21 = Diagram(pool, pool.bool_test(LinearTest("x2", ">=")))
        test22 = Diagram(pool, pool.bool_test(LinearTest("x2", ">")))
        test23 = Diagram(pool, pool.bool_test(LinearTest("x2 - 1", ">")))
        test24 = Diagram(pool, pool.bool_test(LinearTest("x2 - 2", ">")))

        x_twos = test12 * ~test23 * x_two
        twos = test12 * test23 * two
        threes = ~test12 * ~test22 * three
        fours = ~test12 * test22 * four

        unlimited = x_twos + twos + threes + fours
        restricted = unlimited * test11 * ~test13 * test21 * ~test24

        vector = test21 * ~test24 * Diagram(pool, pool.terminal("x2 + 1"))

        result = Diagram(pool, matrix_multiply(pool, restricted.root_node.node_id, vector.root_node.node_id, ["x2"]))
        for x1 in range(0, 4):
            self.assertEqual(8 if x1 < 2 else 23, result.evaluate({"x1": x1}))

if __name__ == '__main__':
    unittest.main()
