import unittest
import os
from pyxadd.build import Builder
from pyxadd.diagram import Diagram, InternalNode, Pool
from pyxadd.order import is_ordered
from pyxadd.test import LinearTest
from pyxadd import walk, leaf_transform
from tests.export import Exporter


class DummyDepthFirstWalker(walk.DepthFirstWalker):
    def __init__(self, diagram):
        walk.DepthFirstWalker.__init__(self, diagram)
        self.numbers = []

    def visit_internal(self, internal_node, parent_message):
        if parent_message is None:
            parent_message = []
        true_message = parent_message + [internal_node.node_id]
        false_message = parent_message + [-internal_node.node_id]
        return true_message, false_message

    def visit_terminal(self, terminal_node, parent_message):
        self.numbers.append((terminal_node.node_id, parent_message))


class DummyDownUpWalker(walk.DownUpWalker):
    def visit_internal_down(self, internal_node, parent_message):
        if parent_message is None:
            parent_message = []
        true_message = parent_message + [internal_node.node_id]
        false_message = parent_message + [-internal_node.node_id]
        return true_message, false_message

    def visit_terminal(self, terminal_node, parent_message):
        return [(terminal_node.node_id, parent_message)]

    def visit_internal_aggregate(self, internal_node, true_result, false_result):
        return true_result + false_result


class DummyBottomUpWalker(walk.BottomUpWalker):
    def visit_terminal(self, terminal_node):
        return terminal_node.node_id

    def visit_internal(self, internal_node, true_message, false_message):
        return true_message - false_message


class Path(object):
    def __init__(self, nodes=list()):
        self.nodes = nodes

    def add(self, node_id):
        return Path(self.nodes + [node_id])

    def __repr__(self):
        return "Path({})".format(", ".join(map(str, self.nodes)))

    def __eq__(self, other):
        return isinstance(other, Path) and self.nodes == other.nodes

    def __hash__(self):
        return hash(tuple(self.nodes))


class DummyTopDownWalker(walk.TopDownWalker):
    def __init__(self, diagram):
        walk.TopDownWalker.__init__(self, diagram)
        self.paths = dict()  # maps node_ids to paths

    def visit_internal(self, internal_node, messages):
        self.register_paths(internal_node.node_id, messages)
        if len(messages) == 0:
            return [Path([internal_node.node_id])], [Path([-internal_node.node_id])]
        result_true = []
        result_false = []
        for paths in messages:
            for path in paths:
                result_true.append(path.add(internal_node.node_id))
                result_false.append(path.add(-internal_node.node_id))
        return result_true, result_false

    def visit_terminal(self, terminal_node, messages):
        self.register_paths(terminal_node.node_id, messages)

    def register_paths(self, node_id, messages):
        self.paths[node_id] = [path for paths in messages for path in paths]


class TestWalking(unittest.TestCase):
    def setUp(self):
        self.diagram = TestWalking.construct_diagram()
        self.exporter = Exporter(os.path.join(os.path.dirname(os.path.realpath(__file__)), "visual"), "walking")

    @staticmethod
    def construct_diagram():
        pool = Pool(empty=True)
        pool.int_var("x")
        x = pool.terminal("x")
        zero = pool.terminal("0")
        test1 = pool.internal(LinearTest("x - 5", "<="), x, zero)
        test2 = pool.internal(LinearTest("x + 1", ">="), test1, zero)
        test3 = pool.internal(LinearTest("x + 2", "<="), x, test2)
        root = pool.internal(LinearTest("x", ">="), test1, test3)
        return Diagram(pool, root)

    @staticmethod
    def build_diagram2():
        build = Builder()
        build.vars("int", "x", "y")

        test1 = build.test("x", "<=", "y")
        test2 = build.test("y", "<=", 5)
        test3 = build.test("x", ">=", 5)

        exp1 = build.exp(5)
        exp2 = build.exp("2 * x")

        node3 = build.ite(test3, exp1, exp2)
        node2 = build.ite(test2, node3, exp1)
        node1 = build.ite(test1, node2, node3)

        if not is_ordered(node1):
            raise RuntimeError("Diagram not ordered")

        return node1

    def test_depth_first(self):
        dummy = DummyDepthFirstWalker(self.diagram)
        dummy.walk()
        self.assertEqual(dummy.numbers, [
            (1, [6, 3]),
            (2, [6, -3]),
            (1, [-6, 5]),
            (1, [-6, -5, 4, 3]),
            (2, [-6, -5, 4, -3]),
            (2, [-6, -5, -4])])

    def test_down_up(self):
        dummy = DummyDownUpWalker(self.diagram)
        result = dummy.walk()
        self.assertEqual(result, [
            (1, [6, 3]),
            (2, [6, -3]),
            (1, [-6, 5]),
            (1, [-6, -5, 4, 3]),
            (2, [-6, -5, 4, -3]),
            (2, [-6, -5, -4])])

    def test_parents_walker(self):
        walker = walk.ParentsWalker(self.diagram)
        parents = walker.walk()
        self.assertEqual(parents, {1: {3, 5}, 2: {3, 4}, 3: {4, 6}, 4: {5}, 5: {6}, 6: set()})

    def test_walking_profile(self):
        ordering = list(walk.WalkingProfile.extract_cache(walk.ParentsWalker(self.diagram).walk(), self.diagram))
        for node_id in [1, 2, 3, 4, 5, 6]:
            node = self.diagram.node(node_id)
            if isinstance(node, InternalNode):
                self_position = ordering.index(node_id)
                children_max = max(ordering.index(node.child_true), ordering.index(node.child_false))
                self.assertTrue(self_position > children_max)

    def test_bottom_up_walker(self):
        profile = walk.WalkingProfile(self.diagram)
        result = DummyBottomUpWalker(self.diagram, profile).walk()
        self.assertEqual(result, -5)

    def test_top_down_walker(self):
        diagram = self.build_diagram2()
        self.exporter.export(diagram, "diagram")
        walker = DummyTopDownWalker(diagram)
        walker.walk()
        control = {
            21: [],
            17: [Path([21])],
            13: [Path([-21]), Path([21, 17])],
            8: [Path([21, -17]), Path([21, 17, 13]), Path([-21, 13])],
            9: [Path([21, 17, -13]), Path([-21, -13])]
        }
        for node_id, paths in control.items():
            self.assertEquals(set(paths), set(walker.paths[node_id]))

    def test_leaf_walking_leaf_node(self):
        pool = Pool()
        diagram = pool.diagram(pool.one_id)
        walk.walk_leaves(lambda p, n: True, diagram)

    def test_leaf_transform_leaf_node(self):
        pool = Pool()
        diagram = pool.diagram(pool.one_id)
        leaf_transform.transform_leaves(lambda t, d: d.pool.terminal(2), diagram)

    def test_leaf_transform_simple_test(self):
        pool = Pool()
        pool.int_var("x")
        test1 = pool.bool_test(LinearTest("x", ">="))
        leaf_transform.transform_leaves(lambda t, d: d.pool.terminal(2), pool.diagram(test1))


if __name__ == '__main__':
    unittest.main()
