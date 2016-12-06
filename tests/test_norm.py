import unittest

from pyxadd import norm
from pyxadd.diagram import Pool, Diagram
from pyxadd.norm import SquareWalker
from pyxadd.test import LinearTest
from pyxadd.view import export


class TestNorm(unittest.TestCase):
    def setUp(self):
        pool = Pool()
        pool.int_var("x")
        lb = Diagram(pool, pool.bool_test(LinearTest("x - 1", ">=")))
        ub = Diagram(pool, pool.bool_test(LinearTest("x - 10", "<=")))
        test = Diagram(pool, pool.bool_test(LinearTest("x - 5", "<=")))
        term_one = Diagram(pool, pool.terminal("x + 2"))
        term_two = Diagram(pool, pool.terminal("7 - 2 * (x - 5)"))

        b1 = lb & ub & test * term_one
        b2 = lb & ub & ~test * term_two

        self.diagram = b1 + b2

    def test_square(self):
        diagram = Diagram(self.diagram.pool, SquareWalker(self.diagram, self.diagram.profile).walk())

        for x in range(0, 12):
            obtained = diagram.evaluate({"x": x})
            if x < 1:
                self.assertEqual(0, obtained)
            elif x <= 5:
                self.assertEqual((x + 2) ** 2, obtained)
            elif x <= 10:
                self.assertEqual((7 - 2 * (x - 5)) ** 2, obtained)
            else:
                self.assertEqual(0, obtained)

    def test_norm(self):
        self.assertTrue(abs(13.41640786 - norm.norm(["x"], self.diagram)) < 0.00000001)
