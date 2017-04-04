import unittest

import experiments.citations.parse as parse
from pyxadd.build import Builder


class TestParse(unittest.TestCase):
    leaf_node_file = "res/parse_leaf.txt"
    simple_tree_file = "res/parse_simple.txt"
    nested_tree_file = "res/parse_nested.txt"

    def setUp(self):
        self.build = Builder()
        self.build.vars("int", "f")

    def test_leaf_node(self):
        truth_function = lambda a: 5
        self.compare_all(self.leaf_node_file, [truth_function])

    def test_leaf_simple(self):
        truth_function = lambda a: 1 if a["f"] <= 5 else 2
        self.compare_all(self.simple_tree_file, [truth_function])

    def test_leaf_nested(self):
        truth_function = lambda a: 1 if a["f"] <= 2 else (4 if a["f"] > 5 else (2 if a["f"] <= 3 else 3))
        self.compare_all(self.nested_tree_file, [truth_function])

    def compare_all(self, filename, truth_functions):
        xadds = parse.read_xadds(filename, self.build.pool)
        self.assertEquals(len(truth_functions), len(xadds))
        eval_functions = parse.read_trees(filename)
        self.assertEquals(len(truth_functions), len(eval_functions))
        for i in range(len(truth_functions)):
            self.compare(xadds[i], eval_functions[i], truth_functions[i])


    def compare(self, xadd, eval_function, truth_function):
        for f in range(10):
            assignment = {"f": f}
            result_truth = truth_function(assignment)
            if xadd is not None:
                result_xadd = xadd.evaluate(assignment)
                self.assertEquals(result_truth, result_xadd)
            if eval_function is not None:
                result_eval = eval_function(assignment)
                self.assertEquals(result_truth, result_eval)
