import unittest

from xadd.diagram import Diagram, Pool
from xadd.matrix_vector import SummationWalker
from xadd.partial import PartialWalker
from xadd.test import Test, Operators
from xadd.view import to_dot, export
from xadd.walk import WalkingProfile


class TestMatrixVector(unittest.TestCase):
    def setUp(self):
        self.diagram = TestMatrixVector.construct_diagram()

    @staticmethod
    def construct_diagram():
        pool = Pool()
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

if __name__ == '__main__':
    unittest.main()
