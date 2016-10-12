import unittest

from pyxadd.diagram import Pool, Diagram
from pyxadd.reduce import LinearReduction, SmtReduce
from pyxadd.test import Test
from pyxadd.view import export


class TestReduce(unittest.TestCase):
    def setUp(self):
        pool = Pool()
        pool.int_var("x")
        lb = Diagram(pool, pool.bool_test(Test("x - 1", ">=")))
        ub = Diagram(pool, pool.bool_test(Test("x - 10", "<=")))
        test = Diagram(pool, pool.bool_test(Test("x - 5", "<=")))
        redundant_test = Diagram(pool, pool.bool_test(Test("x - 6", "<=")))

        term_one = Diagram(pool, pool.terminal("x + 2"))
        term_two = Diagram(pool, pool.terminal("7 - 2 * (x - 5)"))

        b1 = (lb & ub & test & redundant_test) * term_one
        b2 = (lb & ub & ~test & redundant_test) * term_two

        self.diagram = b1 + b2

    def test_reduce(self):
        export(self.diagram, "to_reduce.dot")
        result = LinearReduction(self.diagram.pool).reduce(self.diagram.root_node.node_id, ["x"])
        export(Diagram(self.diagram.pool, result), "result.dot")

    def test_smt_reduce(self):
        export(self.diagram, "to_reduce.dot")
        result = SmtReduce(self.diagram.pool).reduce(self.diagram.root_node.node_id, ["x"])
        export(Diagram(self.diagram.pool, result), "result.dot")
