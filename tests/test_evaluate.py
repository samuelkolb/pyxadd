import random
import unittest

import numpy

from pyxadd.diagram import Pool
from pyxadd.evaluate import mass_evaluate
from pyxadd.timer import Timer


def get_diagram_1():
    pool_file = "data/test_evaluate_1.txt"
    root_id = 1663

    with open(pool_file, "r") as stream:
        json_input = stream.readline()

    exported_pool = Pool.from_json(json_input)
    diagram1 = exported_pool.diagram(root_id)
    vars1 = [('r_f0', 0, 1658), ('r_f1', 0, 964), ('c_f0', 0, 1658), ('c_f1', 0, 964)]
    return diagram1, vars1


class TestDiagram(unittest.TestCase):

    def setUp(self):
        self.diagram1, self.vars1 = get_diagram_1()

    def test_numeric(self):
        entries = []

        timer = Timer()
        timer.start("Creating random assignments")
        for i in range(10000):
            assignment = {}
            for name, lb, ub in self.vars1:
                assignment[name] = random.randint(lb, ub)
            entries.append(assignment)

        timer.start("Mass evaluate assignments")
        evaluated = mass_evaluate(self.diagram1, entries)
        time_mass_evaluation = timer.stop()

        timer.start("Evaluate assignments individually")
        control = []
        for i in range(len(entries)):
            control.append(self.diagram1.evaluate(entries[i]))
        time_individual_evaluation = timer.stop()

        timer.start("Control results")
        for i in range(len(entries)):
            self.assertAlmostEquals(control[i], evaluated[i], delta=10 ** -8)
        timer.stop()

        self.assertTrue(time_mass_evaluation < time_individual_evaluation,
                        "Mass evaluation ({}) slower than individual evaluation ({})"
                        .format(time_mass_evaluation, time_individual_evaluation))
