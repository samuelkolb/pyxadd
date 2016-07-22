import unittest

from pyxadd.diagram import Diagram, Pool
from pyxadd.matrix_vector import SummationWalker, matrix_multiply
from pyxadd.partial import PartialWalker
from pyxadd.test import Test, Operators
from pyxadd.view import export


class TestMatrixVector(unittest.TestCase):
    def setUp(self):
        self.diagram = TestMatrixVector.construct_diagram()

    @staticmethod
    def construct_diagram():
        pool = Pool()
        pool.int_var("x", "y")
        zero = pool.terminal("0")
        f1 = pool.terminal("2*x + 3*y")
        f2 = pool.terminal("3*x + 2*y")
        test0 = pool.internal(Test("x - y", ">="), f1, f2)
        test1 = pool.internal(Test("y - 10", Operators.get("<=")), test0, zero)
        test2 = pool.internal(Test("y - 1", Operators.get(">=")), test1, zero)
        test3 = pool.internal(Test("x - 8", Operators.get("<=")), test2, zero)
        root = pool.internal(Test("x", Operators.get(">=")), test3, zero)
        return Diagram(pool, root)

    def test_mixed_symbolic(self):
        diagram_y = Diagram(self.diagram.pool, SummationWalker(self.diagram, "x").walk())

        result = 0
        for y in range(0, 10):
            result += diagram_y.evaluate({"y": y})

        explicit_result = 0
        for x in range(0, 10):
            for y in range(0, 10):
                explicit_result += self.diagram.evaluate({"x": x, "y": y})

        self.assertEqual(explicit_result, result)

    def test_partial(self):
        partial = PartialWalker(self.diagram, [("y", 2)]).walk()
        for x in range(-10, 10):
            if x < 0 or x > 8:
                self.assertEqual(0, partial.evaluate([("x", x)]))
            elif x > 2:
                self.assertEqual(2 * x + 6, partial.evaluate([("x", x)]))
            else:
                self.assertEqual(3 * x + 4, partial.evaluate([("x", x)]))

    def test_multiplication(self):
        pool = Pool()
        pool.int_var("x1", "x2")
        x_two = Diagram(pool, pool.terminal("x2"))
        two = Diagram(pool, pool.terminal("2"))
        three = Diagram(pool, pool.terminal("3"))
        four = Diagram(pool, pool.terminal("4"))

        test11 = Diagram(pool, pool.bool_test(Test("x1", ">=")))
        test12 = Diagram(pool, pool.bool_test(Test("x1 - 1", "<=")))
        test13 = Diagram(pool, pool.bool_test(Test("x1 - 3", ">")))

        test21 = Diagram(pool, pool.bool_test(Test("x2", ">=")))
        test22 = Diagram(pool, pool.bool_test(Test("x2", ">")))
        test23 = Diagram(pool, pool.bool_test(Test("x2 - 1", ">")))
        test24 = Diagram(pool, pool.bool_test(Test("x2 - 2", ">")))

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
