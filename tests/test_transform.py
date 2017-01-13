from __future__ import print_function

import os
import unittest
from pyxadd.build import Builder
from pyxadd.diagram import Pool
from pyxadd.reduce import LinearReduction
from pyxadd.transform import to_constant, to_ground_tensor
from pyxadd.view import export
from pyxadd.walk import BottomUpWalker, walk_leaves, add_profile_cache, get_profile, map_leaves
from tests.export import Exporter


def is_constant(node):
    return len(node.expression.free_symbols) == 0


class TestTransform(unittest.TestCase):
    def setUp(self):
        pool = Pool()
        self.builder = Builder(pool)
        self.builder.ints("x", "y")

        b = self.builder
        limits = b.limit("x", 0, 20) & b.limit("y", 0, 20)
        rect1 = b.limit("x", 0, 10) & b.limit("y", 0, 10)
        rect2 = b.limit("x", 5, 20) & b.limit("y", 10, 20)
        self.diagram = rect1 * b.terminal("5") \
                  + rect2 * b.terminal("x + 2") \
                  + ~(rect1 | rect2) & limits * b.terminal("1")

        self.exporter = Exporter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"), "transform")

    def test_constant(self):
        b = self.builder
        pool = b.pool

        test_d = b.limit("x", 1, 1) * b.exp(5) + b.limit("x", 2, 2) * b.exp(5)
        test_d = pool.diagram(LinearReduction(pool).reduce(test_d.root_node.node_id))
        self.exporter.export(test_d, "test_d")

        # Test with profile
        add_profile_cache(pool)
        get_profile(self.diagram)

        self.exporter.export(self.diagram, "to_constant")
        constant = to_constant(self.diagram)
        get_profile(constant)

        self.exporter.export(constant, "constant")

        def t(_, node):
            self.assertTrue(is_constant(node), msg="{} still contains variables".format(node.expression))

        walk_leaves(t, constant)

    def test_ground_tensor(self):
        bounds = [("x", 0, 20), ("y", 0, 20)]
        tensor = to_ground_tensor(self.diagram, bounds)
        size = 1
        for s in [ub - lb + 1 for _, lb, ub in bounds]:
            size *= s
        self.assertEqual(len(tensor.flatten()), size)
        for x in range(0, 21):
            for y in range(0, 21):
                self.assertEqual(self.diagram.evaluate({"x": x, "y": y}), tensor[(x, y)])
