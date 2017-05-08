import unittest
import sympy as sym
import os

from pyxadd import build
from pyxadd import order
from pyxadd import rename

from tests import export


# TODO Finish

class RenamingTest(unittest.TestCase):
    def setUp(self):
        b = build.Builder()
        b.ints("x", "a", "b", "c", "d")
        self.builder = b
        a_test = b.test("x", "<=", "a")
        b_test = ~b.test("x", "<=", "b")
        c_test = b.test("x", "<=", "c")
        shared_node = b.ite(b.test("x", "<=", "d"), "3*x + 7", "5*x + 11")
        self.diagram1 = b.ite(a_test,
                              b_test * shared_node,
                              c_test * shared_node)

        self.diagram2 = a_test * b_test * c_test * b.exp("19*x")

        self.exporter = export.Exporter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"), "rename")

    def test_rename_diagram(self):
        translation = {"b": "c", "c": "b"}
        result1 = self.diagram1.pool.diagram(rename.rename(self.diagram1, translation))
        print(order.is_ordered(result1))

        translation = {"b": "c", "c": "b"}
        result2 = self.diagram1.pool.diagram(rename.rename(self.diagram2, translation))
        print(order.is_ordered(result2))

    def test_substitute_diagram(self):
        result = self.diagram1.pool.diagram(rename.substitute(self.diagram1, {"b": "c + 1", "c": "b - 5"}))
