import os
import unittest

from pyxadd import build
from pyxadd.diagram import Pool, Diagram
from pyxadd.matrix import matrix
from pyxadd.reduce import LinearReduction, SmtReduce, SimpleBoundReducer
from pyxadd.test import LinearTest
from pyxadd import walk
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
        # self.exporter.export(self.diagram, "to_reduce")
        result = LinearReduction(self.diagram.pool).reduce(self.diagram.root_node.node_id, ["x"])
        # self.exporter.export(Diagram(self.diagram.pool, result), "result")

    def test_smt_reduce(self):
        # self.exporter.export(self.diagram, "to_reduce")
        result = SmtReduce(self.diagram.pool).reduce(self.diagram.root_node.node_id, ["x"])
        # self.exporter.export(Diagram(self.diagram.pool, result), "result")

    @staticmethod
    def build_diagram_1():
        b = build.Builder()
        b.ints("x")
        test_1 = b.test("x >= 1")
        test_10 = b.test("x <= 10")
        test_5 = b.test("x <= 5")
        test_6 = b.test("x <= 6")

        tree = b.ite(test_5, b.ite(test_6, 3, 5), b.ite(test_6, 7, 11))
        diagram1 = test_1 * test_10 * tree
        return diagram1

    def test_linear_reduce_1(self):
        diagram1 = self.build_diagram_1()
        # self.exporter.export(diagram1, "diagram1")
        variables = [("x", 0, 11)]

        reduced_linear1 = diagram1.pool.diagram(LinearReduction(diagram1.pool).reduce(diagram1.root_id))
        # self.exporter.export(reduced_linear1, "reduced_linear1")
        self.control(variables, diagram1, reduced_linear1)

        reduced_smt1 = diagram1.pool.diagram(SmtReduce(diagram1.pool).reduce(diagram1.root_id))
        # self.exporter.export(reduced_smt1, "reduced_smt1")
        self.control(variables, diagram1, reduced_smt1)

        reduced_simple1 = diagram1.pool.diagram(SimpleBoundReducer(diagram1.pool).reduce(diagram1.root_id))
        # self.exporter.export(reduced_simple1, "reduced_simple1")
        self.control(variables, diagram1, reduced_simple1)

    @staticmethod
    def build_diagram_2():
        b = build.Builder()
        b.ints("x", "y")
        test_1 = b.test("x >= 1") * b.test("y >= 1")
        test_10 = b.test("x <= 10") * b.test("y <= 10")
        test_y = b.test("x <= y")
        test_y_plus_1 = b.test("x <= y + 1")

        tree = b.ite(test_y, b.ite(test_y_plus_1, 3, 5), b.ite(test_y_plus_1, 7, 11))
        diagram1 = test_1 * test_10 * tree
        return diagram1

    def test_linear_reduce_2(self):
        diagram2 = self.build_diagram_2()
        # self.exporter.export(diagram2, "diagram2")
        variables = [("x", 0, 11), ("y", 0, 11)]

        reduced_linear2 = diagram2.pool.diagram(LinearReduction(diagram2.pool).reduce(diagram2.root_id))
        # self.exporter.export(reduced_linear2, "reduced_linear2")
        self.control(variables, diagram2, reduced_linear2)

        reduced_smt2 = diagram2.pool.diagram(SmtReduce(diagram2.pool).reduce(diagram2.root_id))
        # self.exporter.export(reduced_smt2, "reduced_smt2")
        self.control(variables, diagram2, reduced_smt2)

    def control(self, variables, original_diagram, reduced_diagram, allow_unreached=False):
        """
        :type variables: list
        :type original_diagram: Diagram
        :type reduced_diagram: Diagram
        """
        hits = set()
        for assignment in matrix.assignments(variables):
            result_original = original_diagram.evaluate(assignment)
            result_new = reduced_diagram.evaluate(assignment)
            self.assertEqual(result_original, result_new)
            hits.add(result_new)

        present = set()
        walk.walk_leaves(lambda p, t: present.add(t.expression), reduced_diagram.root_id, reduced_diagram.pool)

        if not allow_unreached and len(present - hits) > 0:
            self.fail("Unreached leaves: {}".format(present - hits))
