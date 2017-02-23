import unittest

from pyxadd.diagram import Pool
from pyxadd.order import is_ordered, order
from pyxadd.test import LinearTest
from tests.test_reduction import build_diagram_1


class TestReduce(unittest.TestCase):
    def test_is_ordered_on_ordered_diagram(self):
        diagram = self.get_ordered_diagram()
        self.assertTrue(is_ordered(diagram))

    @staticmethod
    def get_ordered_diagram():
        diagram = build_diagram_1()
        return diagram

    def test_is_ordered_on_looping_diagram(self):
        diagram = self.get_looping_diagram()
        self.assertFalse(is_ordered(diagram))

    @staticmethod
    def get_looping_diagram():
        pool = Pool()
        pool.int_var("x")
        test = LinearTest("x", "<=", "2")
        zero = pool.terminal(0)
        one = pool.terminal(1)
        node1 = pool.internal(test, one, zero)
        node2 = pool.internal(test, node1, zero)
        diagram = pool.diagram(node2)
        return diagram

    def test_is_ordered_on_unordered_diagram(self):
        diagram = self.get_unordered_diagram()
        self.assertFalse(is_ordered(diagram))

    @staticmethod
    def get_unordered_diagram():
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
        return diagram

    def test_reorder_ordered(self):
        ordered = order(self.get_ordered_diagram())
        self.assertTrue(is_ordered(ordered))

    def test_reorder_looping(self):
        ordered = order(self.get_looping_diagram())
        self.assertTrue(is_ordered(ordered))

    def test_reorder_unordered(self):
        ordered = order(self.get_unordered_diagram())
        self.assertTrue(is_ordered(ordered))
