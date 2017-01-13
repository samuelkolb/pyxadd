import os
import unittest

from pyxadd.diagram import Pool, Diagram
from pyxadd.reduce import LinearReduction, SmtReduce
from pyxadd.test import LinearTest
from pyxadd.view import export
from tests.export import Exporter


class TestReduce(unittest.TestCase):
    def setUp(self):
        pool = Pool()
        pool.int_var("x")
        lb = Diagram(pool, pool.bool_test(LinearTest("x - 1", ">=")))
        ub = Diagram(pool, pool.bool_test(LinearTest("x - 10", "<=")))
        test = Diagram(pool, pool.bool_test(LinearTest("x - 5", "<=")))
        redundant_test = Diagram(pool, pool.bool_test(LinearTest("x - 6", "<=")))

        term_one = Diagram(pool, pool.terminal("x + 2"))
        term_two = Diagram(pool, pool.terminal("7 - 2 * (x - 5)"))

        b1 = (lb & ub & test & redundant_test) * term_one
        b2 = (lb & ub & ~test & redundant_test) * term_two

        self.diagram = b1 + b2

        self.exporter = Exporter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"), "reduce")

    def test_reduce(self):
        self.exporter.export(self.diagram, "to_reduce")
        result = LinearReduction(self.diagram.pool).reduce(self.diagram.root_node.node_id, ["x"])
        self.exporter.export(Diagram(self.diagram.pool, result), "result")

    def test_smt_reduce(self):
        self.exporter.export(self.diagram, "to_reduce")
        result = SmtReduce(self.diagram.pool).reduce(self.diagram.root_node.node_id, ["x"])
        self.exporter.export(Diagram(self.diagram.pool, result), "result")
