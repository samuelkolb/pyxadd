import unittest
import sympy as sym

from pyxadd import build
from pyxadd import rename


class RenamingTest(unittest.TestCase):
    def setUp(self):
        b = build.Builder()
        self.builder = b
        shared_node = b.ite(b.test("x", "<=", "d"), "3*x + 7", "5*x + 11")
        self.diagram1 = b.ite(b.test("x", "<=", "a"),
                              ~b.test("x", "<=", "b") * shared_node,
                              b.test("x", "<=", "x") * shared_node)

    def test_rename_diagram(self):
        result = self.diagram1.pool.diagram(rename.rename(self.diagram1, {"b": "c", "c": "b"}))

    def test_substitute_diagram(self):
        result = self.diagram1.pool.diagram(rename.substitute(self.diagram1, {"b": "c + 1", "c": "b - 5"}))
