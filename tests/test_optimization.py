import unittest

from pyxadd.diagram import Pool
from pyxadd.optimization import find_optimal_conditions


class TestIlpOpt(unittest.TestCase):
    def setUp(self):
        pool_file = "data/test_evaluate_1.txt"
        root_id = 1663

        with open(pool_file, "r") as stream:
            json_input = stream.readline()

        exported_pool = Pool.from_json(json_input)
        self.diagram1 = exported_pool.diagram(root_id)
        self.vars1 = [('r_f0', 0, 1658), ('r_f1', 0, 964), ('c_f0', 0, 1658), ('c_f1', 0, 964)]

    def test_optimal_intervals(self):
        print(find_optimal_conditions(self.diagram1, self.vars1))