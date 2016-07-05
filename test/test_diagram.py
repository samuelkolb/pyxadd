import unittest

from xadd.diagram import Diagram, Pool
from xadd.test import Test, Operators


class TestDiagram(unittest.TestCase):
    def setUp(self):
        self.diagram = TestDiagram.construct_diagram()

    @staticmethod
    def construct_diagram():
        pool = Pool(empty=True)
        x = pool.terminal("x")
        zero = pool.terminal("0")
        test1 = pool.internal(Test("x - 5", Operators.get("<=")), x, zero)
        test2 = pool.internal(Test("x + 1", Operators.get(">=")), test1, zero)
        test3 = pool.internal(Test("x + 2", Operators.get("=")), x, test2)
        root = pool.internal(Test("x", Operators.get(">=")), test1, test3)
        return Diagram(pool, root)

    def test_evaluation(self):
        self.assertEqual(4, self.diagram.evaluate([("x", 4)]))


if __name__ == '__main__':
    unittest.main()
