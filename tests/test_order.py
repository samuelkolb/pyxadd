import unittest

from pyxadd.diagram import Pool
from pyxadd.order import is_ordered
from pyxadd.test import LinearTest
from tests.test_reduction import build_diagram_1


class TestReduce(unittest.TestCase):
    def test_is_ordered_on_ordered_diagram(self):
        diagram = build_diagram_1()
        self.assertTrue(is_ordered(diagram))

    def test_is_ordered_on_looping_diagram(self):
        pool = Pool()
        pool.int_var("x")
        test = LinearTest("x", "<=", "2")
        zero = pool.terminal(0)
        one = pool.terminal(1)
        node1 = pool.internal(test, one, zero)
        node2 = pool.internal(test, node1, zero)
        diagram = pool.diagram(node2)
        self.assertFalse(is_ordered(diagram))

    def test_is_ordered_on_unordered_diagram(self):
        pool = Pool()
        pool.int_var("x")
        test1 = LinearTest("x", "<=", "2")
        test2 = LinearTest("x", "<=", "3")

        zero = pool.terminal(0)
        one = pool.terminal(1)

        pool.internal(test1, one, zero)
        pool.internal(test2, one, zero)

        # test2 => test1 => 1
        node1 = pool.internal(test1, one, zero)
        node2 = pool.internal(test2, node1, zero)
        diagram = pool.diagram(node2)

        self.assertFalse(is_ordered(diagram))
