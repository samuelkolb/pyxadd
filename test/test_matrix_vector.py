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
        pool = Pool(empty=True)
        zero = pool.terminal("0")
        f1 = pool.terminal("2*x + 3*y")
        test1 = pool.internal(Test("y - 10", Operators.get("<=")), f1, zero)
        test2 = pool.internal(Test("y", Operators.get(">=")), test1, zero)
        test3 = pool.internal(Test("x - 8", Operators.get("<=")), test2, zero)
        root = pool.internal(Test("x", Operators.get(">=")), test3, zero)
        return Diagram(pool, root)

    def test_mixed_symbolic(self):
        export(self.diagram, "mixed.dot")
        result = SummationWalker(self.diagram, "x").walk()
        print(result)

    def test_partial(self):
        partial = PartialWalker(self.diagram, [("y", 2)]).walk()
        for x in range(-10, 10):
            if x < 0 or x > 8:
                self.assertEqual(0, partial.evaluate([("x", x)]))
            else:
                self.assertEqual(2 * x + 6, partial.evaluate([("x", x)]))


if __name__ == '__main__':
    unittest.main()
