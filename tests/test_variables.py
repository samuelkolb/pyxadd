import unittest

from pyxadd.build import Builder
from pyxadd.diagram import Pool, Diagram
from pyxadd.reduce import LinearReduction, SmtReduce
from pyxadd.test import Test
from pyxadd.variables import VariableFinder
from pyxadd.view import export


class TestVariables(unittest.TestCase):
    def setUp(self):
        pool = Pool()
        pool.int_var("x")
        pool.int_var("y")
        pool.int_var("z")
        b = Builder(pool)

        self.diagrams = []
        d = b.ite(b.test("x", "<=", "y"), b.terminal(1), b.test("x", "<=", 2))
        self.diagrams.append(({"x", "y"}, d))
        d = b.terminal("x * 2 * y + 5 * z")
        self.diagrams.append(({"x", "y", "z"}, d))
        d = b.ite(b.test("x", "<", "y"), b.terminal("z"), b.terminal("z * y"))
        self.diagrams.append(({"x", "y", "z"}, d))

    def test_find_variables(self):
        for variables, diagram in self.diagrams:
            self.assertEqual(variables, VariableFinder(diagram).walk())
