import unittest

from pyxadd import timer
from pyxadd.build import Builder
from pyxadd.diagram import Pool, Diagram
from pyxadd.reduce import LinearReduction, SmtReduce
from pyxadd.test import LinearTest
from pyxadd.variables import VariableFinder, TopDownVariableFinder
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

    def test_find_variables_symbolic_xor(self):
        from tests import test_matrix_vector as mv
        n = 200
        diagram = mv.build_symbolic_xor(n)
        expected_variables = {"x", "c"} | {"c{}".format(i) for i in range(1, n + 1)}

        stop_watch = timer.Timer()
        stop_watch.start("Bottom up")
        self.assertEquals(expected_variables, VariableFinder(diagram).walk())
        stop_watch.start("Top Down")
        self.assertEquals(expected_variables, TopDownVariableFinder(diagram).walk())
        stop_watch.stop()
